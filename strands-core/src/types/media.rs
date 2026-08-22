//! Media content types — images, documents, audio and video.
//!
//! Modelled after the Bedrock API shapes upstream follows, so adapters can map
//! to provider payloads without re-deriving the taxonomy.

use serde::{Deserialize, Serialize};

/// Where a piece of media actually lives.
///
/// Providers accept either inline bytes or a reference to remote storage;
/// carrying both in one type keeps adapters from inventing their own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MediaSource {
    /// Base64-encoded inline content.
    Bytes(String),
    /// A reference to media held elsewhere (e.g. S3).
    Location(SourceLocation),
}

impl MediaSource {
    /// Convenience constructor for inline base64 content.
    pub fn bytes(data: impl Into<String>) -> Self {
        MediaSource::Bytes(data.into())
    }

    /// Convenience constructor for an S3 object.
    pub fn s3(uri: impl Into<String>) -> Self {
        MediaSource::Location(SourceLocation::S3 {
            uri: uri.into(),
            bucket_owner: None,
        })
    }
}

/// A pointer to media stored outside the message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SourceLocation {
    /// An object in an Amazon S3 bucket.
    #[serde(rename = "s3")]
    S3 {
        /// Object URI, starting `s3://`.
        uri: String,
        /// Owning account id, when the bucket belongs to another account.
        #[serde(skip_serializing_if = "Option::is_none")]
        bucket_owner: Option<String>,
    },
    /// A provider-specific location the SDK does not model directly.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Webp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    pub format: ImageFormat,
    pub source: MediaSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    Pdf,
    Csv,
    Doc,
    Docx,
    Xls,
    Xlsx,
    Html,
    Txt,
    Md,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentContent {
    pub format: DocumentFormat,
    /// Human-readable document name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub source: MediaSource,
    /// Whether the model may cite this document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
    /// Extra context supplied alongside the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationsConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Mp3,
    Opus,
    Wav,
    Aac,
    Flac,
    Mp4,
    Ogg,
    Mkv,
    Mka,
    #[serde(rename = "x-aac")]
    XAac,
    M4a,
    Mpeg,
    Mpga,
    Pcm,
    Webm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    pub format: AudioFormat,
    pub source: MediaSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoFormat {
    Flv,
    Mkv,
    Mov,
    Mpeg,
    Mpg,
    Mp4,
    ThreeGp,
    Webm,
    Wmv,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoContent {
    pub format: VideoFormat,
    pub source: MediaSource,
}
