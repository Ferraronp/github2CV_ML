"""Public domain contracts for github2CV ML."""

from github2cv_ml.domain.evidence import EvidenceKind, EvidenceSource, RepoEvidence
from github2cv_ml.domain.profile import (
    CandidateClaim,
    CandidateIdentity,
    CandidateProfile,
    ClaimKind,
)
from github2cv_ml.domain.repository import (
    RepositoryRef,
    RepoSnapshot,
    RepoTreeEntry,
    RepoTreeEntryKind,
)
from github2cv_ml.domain.resume import ResumeDocument, ResumeItem, ResumeSection
from github2cv_ml.domain.signals import (
    DependencyEcosystem,
    DependencySignal,
    ManifestKind,
    ManifestSignal,
    RepositorySignals,
    StructureKind,
    StructureSignal,
    ToolingKind,
    ToolingSignal,
)

__all__ = [
    "CandidateClaim",
    "CandidateIdentity",
    "CandidateProfile",
    "ClaimKind",
    "DependencyEcosystem",
    "DependencySignal",
    "EvidenceKind",
    "EvidenceSource",
    "ManifestKind",
    "ManifestSignal",
    "RepoEvidence",
    "RepoSnapshot",
    "RepoTreeEntry",
    "RepoTreeEntryKind",
    "RepositoryRef",
    "RepositorySignals",
    "ResumeDocument",
    "ResumeItem",
    "ResumeSection",
    "StructureKind",
    "StructureSignal",
    "ToolingKind",
    "ToolingSignal",
]
