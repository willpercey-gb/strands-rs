//! Ready-made plugins.
//!
//! Ported from upstream `vended_plugins/`.

pub mod goal;
pub mod injector;
pub mod offloader;
pub mod skills;

pub use goal::{ContainsJudge, GoalJudge, GoalLoop, GoalOutcome, Verdict};
pub use injector::{ContextInjector, InjectedContent, InjectionPlacement};
pub use offloader::{ContextOffloader, RetrieveOffloadedTool};
pub use skills::{Skill, SkillSet};
