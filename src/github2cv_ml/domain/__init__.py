"""Public domain contracts for github2CV ML."""

from github2cv_ml.domain.evidence import EvidenceKind, EvidenceSource, RepoEvidence
from github2cv_ml.domain.profile import CandidateClaim, CandidateIdentity, CandidateProfile, ClaimKind
from github2cv_ml.domain.repository import (
    RepoSnapshot,
    RepoTreeEntry,
    RepoTreeEntryKind,
    RepositoryRef,
)
from github2cv_ml.domain.resume import ResumeDocument, ResumeItem, ResumeSection

__all__ = [
    "CandidateClaim",
    "CandidateIdentity",
    "CandidateProfile",
    "ClaimKind",
    "EvidenceKind",
    "EvidenceSource",
    "RepoEvidence",
    "RepoSnapshot",
    "RepoTreeEntry",
    "RepoTreeEntryKind",
    "RepositoryRef",
    "ResumeDocument",
    "ResumeItem",
    "ResumeSection",
]
