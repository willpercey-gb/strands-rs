//! Ready-made plugins.
//!
//! Ported from upstream `vended_plugins/`.

pub mod offloader;
pub mod skills;

pub use offloader::{ContextOffloader, RetrieveOffloadedTool};
pub use skills::{Skill, SkillSet};
