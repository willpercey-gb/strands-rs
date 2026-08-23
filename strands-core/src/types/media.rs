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

/// Image formats providers accept.
///
/// Variants name the format and carry no other meaning, so they are
/// not individually documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Webp,
}

/// An image to include in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Encoding of the image data.
    pub format: ImageFormat,
    /// Where the image bytes live.
    pub source: MediaSource,
}

/// Document formats providers accept.
///
/// Variants name the format and carry no other meaning, so they are
/// not individually documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
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

/// A document to include in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentContent {
    /// Encoding of the document data.
    pub format: DocumentFormat,
    /// Human-readable document name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Where the document bytes live.
    pub source: MediaSource,
    /// Whether the model may cite this document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
    /// Extra context supplied alongside the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// Whether a document may be cited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationsConfig {
    /// Whether the model may cite this document in its answer.
    pub enabled: bool,
}

/// Audio formats providers accept.
///
/// Variants name the format and carry no other meaning, so they are
/// not individually documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
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

/// Audio to include in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    /// Encoding of the audio data.
    pub format: AudioFormat,
    /// Where the audio bytes live.
    pub source: MediaSource,
}

/// Video formats providers accept.
///
/// Variants name the format and carry no other meaning, so they are
/// not individually documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(missing_docs)]
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

/// Video to include in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoContent {
    /// Encoding of the video data.
    pub format: VideoFormat,
    /// Where the video bytes live.
    pub source: MediaSource,
}
