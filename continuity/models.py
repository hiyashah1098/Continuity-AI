"""Small data models shared by ingestion, retrieval, and the UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    filename: str
    text: str


@dataclass(frozen=True)
class Chunk:
    filename: str
    text: str
    location: str

    @property
    def label(self) -> str:
        return f"{self.filename} ({self.location})"
