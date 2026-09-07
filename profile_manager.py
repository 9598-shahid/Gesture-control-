"""Loads and saves per-user profiles: the custom-gesture training samples
that seed the KNN classifier, plus which of the default gesture->action
mappings are enabled for that user.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple


class ProfileManager:
    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "default"
        return os.path.join(self.profiles_dir, f"{safe}.json")

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not os.path.exists(path):
            return {"name": name, "knn_samples": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, knn_samples: List[Tuple[List[float], str]]) -> None:
        path = self._path(name)
        data = {
            "name": name,
            "knn_samples": [
                {"features": features, "label": label} for features, label in knn_samples
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def hydrate_knn(knn_classifier, profile_data: dict) -> int:
        """Populates a KNNClassifier (native or fallback) from a loaded
        profile dict. Returns the number of samples loaded."""
        knn_classifier.clear()
        count = 0
        for entry in profile_data.get("knn_samples", []):
            knn_classifier.add_sample(entry["features"], entry["label"])
            count += 1
        return count
