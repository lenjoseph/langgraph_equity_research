"""SQLite-backed checkpoint saver compatible with the installed LangGraph checkpoint API."""

import pickle
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver


class SqliteSnapshotSaver(InMemorySaver):
    """In-memory checkpoint saver that commits each write to a SQLite snapshot."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoint_snapshot (id INTEGER PRIMARY KEY, payload BLOB)"
        )
        self._conn.commit()
        self._load()

    def _load(self) -> None:
        row = self._conn.execute(
            "SELECT payload FROM checkpoint_snapshot WHERE id = 1"
        ).fetchone()
        if not row or not row[0]:
            return
        payload = pickle.loads(row[0])
        storage: defaultdict = defaultdict(lambda: defaultdict(dict))
        for thread_id, namespaces in payload["storage"].items():
            for namespace, checkpoints in namespaces.items():
                storage[thread_id][namespace].update(checkpoints)
        writes: defaultdict = defaultdict(dict)
        for key, value in payload["writes"].items():
            writes[key].update(value)
        self.storage = storage
        self.writes = writes
        self.blobs = payload["blobs"]

    def _persist(self) -> None:
        payload = pickle.dumps(
            {
                "storage": {
                    thread_id: {
                        namespace: dict(checkpoints)
                        for namespace, checkpoints in namespaces.items()
                    }
                    for thread_id, namespaces in self.storage.items()
                },
                "writes": {key: dict(value) for key, value in self.writes.items()},
                "blobs": dict(self.blobs),
            }
        )
        self._conn.execute(
            "INSERT INTO checkpoint_snapshot (id, payload) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
            (payload,),
        )
        self._conn.commit()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._lock:
            saved = super().put(config, checkpoint, metadata, new_versions)
            self._persist()
            return saved

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._persist()
