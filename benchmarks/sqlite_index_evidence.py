"""Opt-in index removal experiments on disposable databases, never production stores."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from async_batch_llm import (
    ArtifactIdentity,
    CallableStrategy,
    CallOutcome,
    LLMWorkItem,
    ResumePolicy,
    SqliteArtifactStore,
    WorkItemResult,
)

INDEXES = {
    "full": None,
    "without_success": "idx_item_records_replay_success",
    "without_all": "idx_item_records_replay_all",
    "without_sequence": "idx_item_records_success_sequence",
}


async def experiment(directory, workload, variant, keys, versions, trial):
    async def invoke(prompt):
        return CallOutcome(output=prompt)

    strategy = CallableStrategy(invoke)
    path = directory / "run.sqlite"
    store = SqliteArtifactStore(path, identity=ArtifactIdentity(provider="evidence", model="fake"))
    items = [
        LLMWorkItem(str(i), strategy, "p", context=None if i % 2 == 0 else {"key": i})
        for i in range(keys)
    ]
    prepared = [await store.prepare_item(item) for item in items]
    await store.append(
        items[0], prepared[0], WorkItemResult(item_id="0", success=True, output="seed")
    )

    def measure():
        connection = store._require_connection()
        template = dict(connection.execute("SELECT * FROM item_records").fetchone())
        schema = [
            tuple(row)
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND name LIKE 'idx_item_records_%'"
            )
        ]
        connection.execute("DELETE FROM item_records")
        connection.commit()
        if INDEXES[variant]:
            connection.execute(f"DROP INDEX {INDEXES[variant]}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # Retain the connection's production synchronous and checkpoint settings.
        pragmas = {
            name: connection.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("journal_mode", "synchronous", "wal_autocheckpoint", "page_size")
        }
        columns = [name for name in template if name != "record_sequence"]
        insert = f"INSERT INTO item_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
        records = []
        expected = {
            policy.value: {} for policy in (ResumePolicy.REUSE_ALL, ResumePolicy.REUSE_SUCCESSES)
        }
        for version in range(versions):
            for item, key in zip(items, prepared, strict=True):
                success = (
                    version % 10 != 0
                    if workload == "success_heavy"
                    else version % 100 == 0
                    if workload == "failure_heavy"
                    else version == 0
                )
                # Last two versions exercise explicit ineligibility and legacy
                # category exclusion even when the stored eligibility bit is 1.
                excluded = version >= versions - 2
                success = success and not excluded
                category = (
                    "batch_aborted" if version == versions - 1 else None if success else "unknown"
                )
                eligible = int(version != versions - 2)
                row = dict(template)
                payload = json.loads(template["result_json"])
                payload.update(
                    item_id=item.item_id,
                    success=success,
                    output=f"value-{version}" if success else None,
                    error=None if success else "fixture failure",
                    error_category=category,
                )
                row.update(
                    item_id=item.item_id,
                    prompt_fingerprint=key.prompt_fingerprint,
                    context_fingerprint=key.context_fingerprint,
                    input_fingerprint=key.input_fingerprint,
                    success=int(success),
                    error_category=category,
                    replay_eligible=eligible,
                    result_json=json.dumps(payload),
                )
                records.append(tuple(row[name] for name in columns))
                if not excluded:
                    expected[ResumePolicy.REUSE_ALL.value][item.item_id] = (success, version)
                    if success:
                        expected[ResumePolicy.REUSE_SUCCESSES.value][item.item_id] = (True, version)
        insert_seconds = commit_seconds = 0.0
        for start in range(0, len(records), 500):
            connection.execute("BEGIN IMMEDIATE")
            clock = time.perf_counter()
            connection.executemany(insert, records[start : start + 500])
            insert_seconds += time.perf_counter() - clock
            clock = time.perf_counter()
            connection.commit()
            commit_seconds += time.perf_counter() - clock
        sizes = {
            "db_bytes": path.stat().st_size,
            "wal_bytes": Path(str(path) + "-wal").stat().st_size,
            "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
        }
        results = []
        identity = template["identity_fingerprint"]
        for analyzed in (False, True):
            if analyzed:
                connection.execute("ANALYZE")
                connection.commit()
            for policy in (ResumePolicy.REUSE_ALL, ResumePolicy.REUSE_SUCCESSES):
                timings = []
                digest = hashlib.sha256()
                plans = []
                for item, key in zip(items, prepared, strict=True):
                    # Exercise production query, row validation, and legacy exclusion.
                    clock = time.perf_counter()
                    record = store._lookup_sync(item.item_id, key, identity, policy)
                    timings.append(time.perf_counter() - clock)
                    assert record is not None
                    success, version = expected[policy.value][item.item_id]
                    assert record["success"] == success
                    if success:
                        assert record["result"]["output"] == f"value-{version}"
                    # Sequence pins failed versions too; seed sequence is 1.
                    assert record["record_sequence"] == 2 + version * keys + int(item.item_id)
                    digest.update(
                        str((item.item_id, record["record_sequence"], record["success"])).encode()
                    )
                    if len(plans) < 2:
                        extra = " AND success=1" if policy is ResumePolicy.REUSE_SUCCESSES else ""
                        sql = f"""SELECT record_sequence,logical_schema_version,item_id,prompt_fingerprint,
                        context_fingerprint,input_fingerprint,identity_fingerprint,success,error_category,result_json
                        FROM item_records WHERE identity_fingerprint=? AND item_id=? AND prompt_fingerprint=?
                        AND context_fingerprint IS ? AND input_fingerprint=? AND replay_eligible=1{extra}
                        ORDER BY record_sequence DESC"""
                        plans.append(
                            [
                                tuple(row)
                                for row in connection.execute(
                                    "EXPLAIN QUERY PLAN " + sql,
                                    (
                                        identity,
                                        item.item_id,
                                        key.prompt_fingerprint,
                                        key.context_fingerprint,
                                        key.input_fingerprint,
                                    ),
                                )
                            ]
                        )
                miss_timings = []
                for _ in range(10):
                    clock = time.perf_counter()
                    assert store._lookup_sync("missing", prepared[0], identity, policy) is None
                    miss_timings.append(time.perf_counter() - clock)
                results.append(
                    {
                        "analyzed": analyzed,
                        "policy": policy.value,
                        "lookup_seconds": timings,
                        "median_seconds": statistics.median(timings),
                        "missing_lookup_seconds": miss_timings,
                        "digest": digest.hexdigest(),
                        "plans_null_and_nonnull": plans,
                    }
                )
            clock = time.perf_counter()
            rows = store._read_page_sync(-1, keys * versions + 1, True)
            elapsed = time.perf_counter() - clock
            assert rows and all(row["success"] for row in rows)
            results.append(
                {
                    "analyzed": analyzed,
                    "success_page_seconds": elapsed,
                    "page_rows": len(rows),
                    "page_plan": [
                        tuple(row)
                        for row in connection.execute(
                            "EXPLAIN QUERY PLAN SELECT * FROM item_records WHERE record_sequence>? "
                            "AND record_sequence<=? AND success=1 ORDER BY record_sequence LIMIT ?",
                            (-1, keys * versions + 1, store.read_batch_size),
                        )
                    ],
                }
            )
        return {
            "workload": workload,
            "variant": variant,
            "trial": trial,
            "keys": keys,
            "versions": versions,
            "rows": len(records),
            "indexes": schema,
            "pragmas": pragmas,
            "insert_seconds": insert_seconds,
            "commit_seconds": commit_seconds,
            **sizes,
            "measurements": results,
        }

    try:
        return await store._run_db(measure)
    finally:
        await store.close()


async def main(args):
    results = {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "runs": []}
    for trial in range(args.trials):
        variants = list(INDEXES)
        variants = variants[trial % 4 :] + variants[: trial % 4]
        for workload in ("success_heavy", "failure_heavy", "old_success"):
            for variant in variants:
                with tempfile.TemporaryDirectory(prefix="abl-e-index-") as directory:
                    results["runs"].append(
                        await experiment(
                            Path(directory), workload, variant, args.keys, args.versions, trial
                        )
                    )
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    # All physical designs must return identical rows under each replay policy.
    digests = {}
    for run in results["runs"]:
        for row in run["measurements"]:
            if "digest" in row:
                key = run["workload"], row["policy"]
                assert digests.setdefault(key, row["digest"]) == row["digest"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=60)
    parser.add_argument("--versions", type=int, default=300)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
