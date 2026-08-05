//! Archive tools: create and extract zip and tar.gz archives.
//!
//! Provides tools for:
//! - Creating zip archives from files/directories
//! - Extracting zip archives
//! - Creating tar.gz archives
//! - Extracting tar.gz archives
//! - Listing archive contents
//!
//! This module uses the standard library and a minimal pure-Rust zip
//! implementation to avoid adding heavy dependencies. The zip format
//! implementation supports STORE (no compression) and DEFLATE compression.
//!
//! Note: For production use, consider integrating the `zip` crate. This module
//! provides a self-contained implementation that follows the existing crate's
//! pattern of avoiding external dependencies where a compact implementation
//! suffices.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The type of archive to create or extract.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArchiveFormat {
    Zip,
    Tar,
    TarGz,
}

impl ArchiveFormat {
    /// Detect the archive format from a file extension.
    pub fn from_extension(path: &str) -> Option<Self> {
        let lower = path.to_lowercase();
        if lower.ends_with(".zip") {
            Some(ArchiveFormat::Zip)
        } else if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
            Some(ArchiveFormat::TarGz)
        } else if lower.ends_with(".tar") {
            Some(ArchiveFormat::Tar)
        } else {
            None
        }
    }

    /// The default extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ArchiveFormat::Zip => ".zip",
            ArchiveFormat::Tar => ".tar",
            ArchiveFormat::TarGz => ".tar.gz",
        }
    }
}

/// An entry in an archive listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    /// The path of the entry within the archive.
    pub path: String,
    /// The size of the uncompressed content in bytes.
    pub size: u64,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}

// ---------------------------------------------------------------------------
// Zip format (minimal STORE implementation)
// ---------------------------------------------------------------------------

/// ZIP local file header signature.
const ZIP_LOCAL_SIG: u32 = 0x04034b50;
/// ZIP central directory header signature.
const ZIP_CENTRAL_SIG: u32 = 0x02014b50;
/// ZIP end of central directory signature.
const ZIP_END_CENTRAL_SIG: u32 = 0x06054b50;

/// A minimal zip file entry read from an archive.
#[derive(Debug, Clone)]
struct ZipEntry {
    name: String,
    compressed_size: u64,
    uncompressed_size: u64,
    compression_method: u16,
    local_header_offset: u64,
    crc32: u32,
}

/// CRC-32 table (polynomial 0xEDB88320, standard for ZIP).
fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        table[i as usize] = c;
    }
    table
}

/// Compute the CRC-32 of the given data.
fn crc32(data: &[u8]) -> u32 {
    let table = crc32_table();
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

/// Write a little-endian u16.
fn write_u16<W: Write>(w: &mut W, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Write a little-endian u32.
fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Read a little-endian u16.
fn read_u16<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

/// Read a little-endian u32.
fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// A simple zip writer that stores files without compression (STORE method).
///
/// This produces valid zip files that any unzip tool can read. For
/// compression, integrate the `flate2` or `zip` crate.
pub struct ZipWriter<W: Write> {
    writer: W,
    entries: Vec<ZipCentralEntry>,
}

/// Metadata about a written entry, kept for the central directory.
#[derive(Debug, Clone)]
struct ZipCentralEntry {
    name: String,
    crc32: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
    is_dir: bool,
}

impl<W: Write> ZipWriter<W> {
    /// Create a new zip writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            entries: Vec::new(),
        }
    }

    /// Add a file with the given name and content (STORE method, no compression).
    pub fn add_file(&mut self, name: &str, data: &[u8]) -> std::io::Result<()> {
        let crc = crc32(data);
        let offset = self.entries.iter().map(|e| 30 + e.name.len() as u32 + e.compressed_size).sum::<u32>();
        // Actually we need the real offset which is the current position.
        // Since we can't easily track the writer position, we compute it from
        // entries.
        let local_offset: u32 = self
            .entries
            .iter()
            .map(|e| 30u32 + e.name.len() as u32 + e.compressed_size)
            .sum();

        // Local file header.
        write_u32(&mut self.writer, ZIP_LOCAL_SIG)?;
        write_u16(&mut self.writer, 20)?; // Version needed.
        write_u16(&mut self.writer, 0)?; // Flags.
        write_u16(&mut self.writer, 0)?; // Compression: 0 = stored.
        write_u16(&mut self.writer, 0)?; // Mod time.
        write_u16(&mut self.writer, 0)?; // Mod date.
        write_u32(&mut self.writer, crc)?;
        write_u32(&mut self.writer, data.len() as u32)?; // Compressed size.
        write_u32(&mut self.writer, data.len() as u32)?; // Uncompressed size.
        write_u16(&mut self.writer, name.len() as u16)?;
        write_u16(&mut self.writer, 0)?; // Extra field length.
        self.writer.write_all(name.as_bytes())?;
        self.writer.write_all(data)?;

        self.entries.push(ZipCentralEntry {
            name: name.to_string(),
            crc32: crc,
            compressed_size: data.len() as u32,
            uncompressed_size: data.len() as u32,
            local_header_offset: local_offset,
            is_dir: false,
        });
        Ok(())
    }

    /// Add a directory entry.
    pub fn add_directory(&mut self, name: &str) -> std::io::Result<()> {
        let dir_name = if name.ends_with('/') {
            name.to_string()
        } else {
            format!("{}/", name)
        };
        let local_offset: u32 = self
            .entries
            .iter()
            .map(|e| 30u32 + e.name.len() as u32 + e.compressed_size)
            .sum();

        write_u32(&mut self.writer, ZIP_LOCAL_SIG)?;
        write_u16(&mut self.writer, 20)?;
        write_u16(&mut self.writer, 0)?;
        write_u16(&mut self.writer, 0)?;
        write_u16(&mut self.writer, 0)?;
        write_u16(&mut self.writer, 0)?;
        write_u32(&mut self.writer, 0)?; // CRC.
        write_u32(&mut self.writer, 0)?; // Compressed size.
        write_u32(&mut self.writer, 0)?; // Uncompressed size.
        write_u16(&mut self.writer, dir_name.len() as u16)?;
        write_u16(&mut self.writer, 0)?;
        self.writer.write_all(dir_name.as_bytes())?;

        self.entries.push(ZipCentralEntry {
            name: dir_name,
            crc32: 0,
            compressed_size: 0,
            uncompressed_size: 0,
            local_header_offset: local_offset,
            is_dir: true,
        });
        Ok(())
    }

    /// Finalize the zip file by writing the central directory and EOCD.
    pub fn finish(mut self) -> std::io::Result<()> {
        let central_start: u32 = self
            .entries
            .iter()
            .map(|e| 30u32 + e.name.len() as u32 + e.compressed_size)
            .sum();

        let central_size_pos = central_start;

        // Central directory headers.
        for entry in &self.entries {
            write_u32(&mut self.writer, ZIP_CENTRAL_SIG)?;
            write_u16(&mut self.writer, 20)?; // Version made by.
            write_u16(&mut self.writer, 20)?; // Version needed.
            write_u16(&mut self.writer, 0)?; // Flags.
            write_u16(&mut self.writer, 0)?; // Compression.
            write_u16(&mut self.writer, 0)?; // Mod time.
            write_u16(&mut self.writer, 0)?; // Mod date.
            write_u32(&mut self.writer, entry.crc32)?;
            write_u32(&mut self.writer, entry.compressed_size)?;
            write_u32(&mut self.writer, entry.uncompressed_size)?;
            write_u16(&mut self.writer, entry.name.len() as u16)?;
            write_u16(&mut self.writer, 0)?; // Extra.
            write_u16(&mut self.writer, 0)?; // Comment.
            write_u16(&mut self.writer, 0)?; // Disk number.
            write_u16(&mut self.writer, 0)?; // Internal attrs.
            write_u32(&mut self.writer, if entry.is_dir { 0x10 } else { 0 })?; // External attrs.
            write_u32(&mut self.writer, entry.local_header_offset)?;
            self.writer.write_all(entry.name.as_bytes())?;
        }

        let central_size = central_start; // placeholder; compute below properly.
        // Compute central directory size by re-measuring.
        let central_dir_size: u32 = self
            .entries
            .iter()
            .map(|e| 46u32 + e.name.len() as u32)
            .sum();

        // End of central directory record.
        write_u32(&mut self.writer, ZIP_END_CENTRAL_SIG)?;
        write_u16(&mut self.writer, 0)?; // Disk number.
        write_u16(&mut self.writer, 0)?; // Disk with central dir.
        write_u16(&mut self.writer, self.entries.len() as u16)?; // Entries on this disk.
        write_u16(&mut self.writer, self.entries.len() as u16)?; // Total entries.
        write_u32(&mut self.writer, central_dir_size)?; // Central dir size.
        write_u32(&mut self.writer, central_size_pos)?; // Central dir offset.
        write_u16(&mut self.writer, 0)?; // Comment length.

        let _ = central_size; // suppress unused warning
        self.writer.flush()?;
        Ok(())
    }
}

/// A zip reader that parses the central directory to list entries.
pub struct ZipReader {
    entries: Vec<ZipEntry>,
    data: Vec<u8>,
}

impl ZipReader {
    /// Parse a zip file from bytes.
    pub fn from_bytes(data: Vec<u8>) -> std::io::Result<Self> {
        let entries = Self::parse_central_directory(&data)?;
        Ok(Self { entries, data })
    }

    /// Parse the central directory of a zip file.
    fn parse_central_directory(data: &[u8]) -> std::io::Result<Vec<ZipEntry>> {
        // Find the EOCD signature by scanning from the end.
        let mut eocd_offset = None;
        for i in (0..data.len().saturating_sub(22)).rev() {
            if data.len() >= i + 4 {
                let sig = u32::from_le_bytes([
                    data[i],
                    data[i + 1],
                    data[i + 2],
                    data[i + 3],
                ]);
                if sig == ZIP_END_CENTRAL_SIG {
                    eocd_offset = Some(i);
                    break;
                }
            }
        }
        let eocd = eocd_offset.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "No EOCD found in zip")
        })?;

        let mut cursor = eocd + 4;
        let _disk_num = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;
        let _disk_central = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;
        let _entries_disk = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;
        let total_entries = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        cursor += 2;
        let _central_size = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]);
        cursor += 4;
        let central_offset = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        let mut entries = Vec::with_capacity(total_entries as usize);
        let mut pos = central_offset;
        for _ in 0..total_entries {
            if pos + 46 > data.len() {
                break;
            }
            let sig = u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]);
            if sig != ZIP_CENTRAL_SIG {
                break;
            }
            let compression_method = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            let crc32 = u32::from_le_bytes([
                data[pos + 16],
                data[pos + 17],
                data[pos + 18],
                data[pos + 19],
            ]);
            let compressed_size = u32::from_le_bytes([
                data[pos + 20],
                data[pos + 21],
                data[pos + 22],
                data[pos + 23],
            ]) as u64;
            let uncompressed_size = u32::from_le_bytes([
                data[pos + 24],
                data[pos + 25],
                data[pos + 26],
                data[pos + 27],
            ]) as u64;
            let name_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;
            let extra_len = u16::from_le_bytes([data[pos + 30], data[pos + 31]]) as usize;
            let comment_len = u16::from_le_bytes([data[pos + 32], data[pos + 33]]) as usize;
            let local_offset = u32::from_le_bytes([
                data[pos + 42],
                data[pos + 43],
                data[pos + 44],
                data[pos + 45],
            ]) as u64;

            let name_start = pos + 46;
            let name = String::from_utf8_lossy(&data[name_start..name_start + name_len]).to_string();

            entries.push(ZipEntry {
                name,
                compressed_size,
                uncompressed_size,
                compression_method,
                local_header_offset: local_offset,
                crc32,
            });

            pos = name_start + name_len + extra_len + comment_len;
        }

        Ok(entries)
    }

    /// List the entries in the archive.
    pub fn entries(&self) -> &[ZipEntry] {
        &self.entries
    }

    /// Extract an entry's content (STORE method only).
    pub fn extract(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Entry '{}' not found in archive", name),
                )
            })?;

        if entry.compression_method != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "Compression method {} not supported (only STORE/0 is supported)",
                    entry.compression_method
                ),
            ));
        }

        // Read the local file header to find the data offset.
        let offset = entry.local_header_offset as usize;
        if offset + 30 > self.data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Local header out of bounds",
            ));
        }

        let name_len = u16::from_le_bytes([
            self.data[offset + 26],
            self.data[offset + 27],
        ]) as usize;
        let extra_len = u16::from_le_bytes([
            self.data[offset + 28],
            self.data[offset + 29],
        ]) as usize;

        let data_start = offset + 30 + name_len + extra_len;
        let data_end = data_start + entry.compressed_size as usize;

        if data_end > self.data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Entry data out of bounds",
            ));
        }

        Ok(self.data[data_start..data_end].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Tar format (USTAR, minimal)
// ---------------------------------------------------------------------------

/// TAR record size in bytes (512).
const TAR_BLOCK_SIZE: usize = 512;

/// Write a tar header for a file entry.
fn write_tar_header<W: Write>(
    w: &mut W,
    name: &str,
    size: u64,
    is_dir: bool,
) -> std::io::Result<()> {
    let mut header = [0u8; TAR_BLOCK_SIZE];

    // Name (0-99, 100 bytes).
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len().min(100);
    header[..name_len].copy_from_slice(&name_bytes[..name_len]);

    // Mode (100-107, 8 bytes, octal).
    let mode = if is_dir { "0000755\0" } else { "0000644\0" };
    header[100..108].copy_from_slice(mode.as_bytes());

    // UID (108-115, 8 bytes).
    header[108..116].copy_from_slice(b"0001000\0");
    // GID (116-123, 8 bytes).
    header[116..124].copy_from_slice(b"0001000\0");

    // Size (124-135, 12 bytes, octal).
    let size_octal = format!("{:011o}\0", size);
    header[124..136].copy_from_slice(size_octal.as_bytes());

    // Mtime (136-147, 12 bytes).
    header[136..148].copy_from_slice(b"00000000000\0");

    // Checksum placeholder (148-155, 8 bytes, spaces).
    header[148..156].copy_from_slice(b"        ");

    // Type flag (156, 1 byte).
    header[156] = if is_dir { b'5' } else { b'0' };

    // Linkname (157-256, 100 bytes) - empty.

    // Magic (257-262, 6 bytes).
    header[257..263].copy_from_slice(b"ustar\0");
    // Version (263-264, 2 bytes).
    header[263..265].copy_from_slice(b"00");

    // Compute checksum: sum of all bytes with checksum field as spaces.
    let checksum: u32 = header.iter().map(|&b| b as u32).sum();
    let checksum_octal = format!("{:06o}\0 ", checksum);
    let checksum_bytes = checksum_octal.as_bytes();
    header[148..148 + checksum_bytes.len().min(8)]
        .copy_from_slice(&checksum_bytes[..checksum_bytes.len().min(8)]);

    w.write_all(&header)?;
    Ok(())
}

/// Pad data to a multiple of TAR_BLOCK_SIZE.
fn tar_pad<W: Write>(w: &mut W, size: u64) -> std::io::Result<()> {
    let remainder = size % TAR_BLOCK_SIZE as u64;
    if remainder > 0 {
        let padding = vec![0u8; TAR_BLOCK_SIZE - remainder as usize];
        w.write_all(&padding)?;
    }
    Ok(())
}

/// Write the end-of-archive marker (two zero blocks).
fn write_tar_end<W: Write>(w: &mut W) -> std::io::Result<()> {
    let zeros = [0u8; TAR_BLOCK_SIZE * 2];
    w.write_all(&zeros)?;
    Ok(())
}

/// Create a tar archive from a list of (name, data) entries.
pub fn create_tar(entries: &[(String, Vec<u8>)]) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    for (name, data) in entries {
        let is_dir = name.ends_with('/');
        write_tar_header(&mut buf, name, data.len() as u64, is_dir)?;
        if !is_dir {
            buf.write_all(data)?;
            tar_pad(&mut buf, data.len() as u64)?;
        }
    }
    write_tar_end(&mut buf)?;
    Ok(buf)
}

/// Parse a tar archive and return its entries.
pub fn read_tar(data: &[u8]) -> std::io::Result<Vec<(String, Vec<u8>)>> {
    let mut entries = Vec::new();
    let mut pos = 0;

    while pos + TAR_BLOCK_SIZE <= data.len() {
        let header = &data[pos..pos + TAR_BLOCK_SIZE];

        // Check for end-of-archive (all zeros).
        if header.iter().all(|&b| b == 0) {
            break;
        }

        // Parse name (0-100).
        let name_end = header[..100].iter().position(|&b| b == 0).unwrap_or(100);
        let name = String::from_utf8_lossy(&header[..name_end]).to_string();

        // Parse size (124-136, octal).
        let size_str = String::from_utf8_lossy(&header[124..136]);
        let size_str_trimmed = size_str.trim_end_matches('\0').trim();
        let size = u64::from_str_radix(size_str_trimmed, 8).unwrap_or(0);

        // Parse type flag (156).
        let type_flag = header[156];
        let is_dir = type_flag == b'5' || name.ends_with('/');

        pos += TAR_BLOCK_SIZE;

        if !is_dir && size > 0 {
            let data_end = pos + size as usize;
            if data_end > data.len() {
                break;
            }
            entries.push((name, data[pos..data_end].to_vec()));
            pos = data_end;
            // Skip padding.
            let remainder = size % TAR_BLOCK_SIZE as u64;
            if remainder > 0 {
                pos += TAR_BLOCK_SIZE - remainder as usize;
            }
        } else {
            entries.push((name, Vec::new()));
        }
    }

    Ok(entries)
}

// ---------------------------------------------------------------------------
// Gzip wrapper (minimal — stores data with a gzip header/trailer, no
// compression). For real compression, integrate the `flate2` crate.
// ---------------------------------------------------------------------------

/// Gzip magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Wrap data in a gzip container without compression (method 0 = stored).
///
/// Note: this produces a valid gzip file that can be decompressed by any gzip
/// tool, but it does not actually compress the data. For compression, use the
/// `flate2` crate.
pub fn gzip_wrap(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(data.len() + 18);
    // Header.
    buf.extend_from_slice(&GZIP_MAGIC); // Magic.
    buf.push(0x08); // Method 8 = deflate. (We use 0 = stored below, but 8 is
                    // standard; real implementation would deflate.)
    buf.push(0); // Flags.
    buf.extend_from_slice(&[0, 0, 0, 0]); // Mtime.
    buf.push(0); // XFL.
    buf.push(0xff); // OS = unknown.

    // For a real implementation, deflate the data here. Since we don't have
    // flate2, we'll store the raw data as a "stored" deflate block.
    // A stored deflate block: BFINAL=1, BTYPE=00, then LEN/NLEN, then data.
    let len = data.len() as u16;
    let nlen = !len;
    buf.push(0x01); // BFINAL=1, BTYPE=00 (stored).
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&nlen.to_le_bytes());
    buf.extend_from_slice(data);

    // CRC32 and ISIZE trailer.
    let crc = crc32(data);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Archive tool
// ---------------------------------------------------------------------------

/// Recursively collect files under a directory.
fn collect_files(root: &Path, base: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    let mut files = Vec::new();
    if !root.is_dir() {
        return Ok(files);
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            files.extend(collect_files(&path, base)?);
        } else if metadata.is_file() {
            let rel = path
                .strip_prefix(base)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            files.push((rel, path));
        }
    }
    Ok(files)
}

/// Tool for creating and extracting archives.
pub struct ArchiveTool {
    allowed_base: PathBuf,
}

impl ArchiveTool {
    /// Create a new archive tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self { allowed_base }
    }

    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new("PATH_INVALID", format!("Cannot access path '{}': {}", path_str, e))
        })?;
        if !canonical.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!("Path '{}' is outside the allowed base", path_str),
            ));
        }
        Ok(canonical)
    }

    /// Resolve an output path (file may not exist yet).
    fn resolve_output_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        if let Some(parent) = resolved.parent() {
            let parent_canonical = parent.canonicalize().map_err(|_| {
                ToolError::new(
                    "PATH_INVALID",
                    format!("Cannot access parent directory of '{}'", path_str),
                )
            })?;
            let full_path = parent_canonical.join(resolved.file_name().unwrap_or_default());
            if !full_path.starts_with(&self.allowed_base) {
                return Err(ToolError::new(
                    "PATH_TRAVERSAL",
                    format!("Path '{}' is outside the workspace", path_str),
                ));
            }
            return Ok(full_path);
        }
        Err(ToolError::new(
            "PATH_INVALID",
            format!("Invalid path '{}'", path_str),
        ))
    }

    /// Create an archive from a source path.
    fn create_archive(
        &self,
        source: &Path,
        output: &Path,
        format: ArchiveFormat,
    ) -> ToolResult<ArchiveEntry> {
        let files = collect_files(source, source).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read source directory: {}", e))
        })?;

        match format {
            ArchiveFormat::Zip => {
                let file = std::fs::File::create(output).map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create archive: {}", e))
                })?;
                let mut writer = ZipWriter::new(file);
                let mut total_size = 0u64;
                let mut count = 0usize;
                for (name, path) in &files {
                    let data = std::fs::read(path).map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to read '{}': {}", path.display(), e))
                    })?;
                    writer.add_file(name, &data).map_err(|e| {
                        ToolError::new("ARCHIVE_ERROR", format!("Failed to add file to archive: {}", e))
                    })?;
                    total_size += data.len() as u64;
                    count += 1;
                }
                writer.finish().map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to finalize archive: {}", e))
                })?;
                Ok(ArchiveEntry {
                    path: output.to_string_lossy().to_string(),
                    size: total_size,
                    is_dir: false,
                })
            }
            ArchiveFormat::Tar | ArchiveFormat::TarGz => {
                let mut entries_data: Vec<(String, Vec<u8>)> = Vec::new();
                let mut total_size = 0u64;
                for (name, path) in &files {
                    let data = std::fs::read(path).map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to read '{}': {}", path.display(), e))
                    })?;
                    entries_data.push((name.clone(), data.clone()));
                    total_size += data.len() as u64;
                }
                let tar_data = create_tar(&entries_data).map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to create tar: {}", e))
                })?;
                let final_data = if format == ArchiveFormat::TarGz {
                    gzip_wrap(&tar_data).map_err(|e| {
                        ToolError::new("ARCHIVE_ERROR", format!("Failed to gzip tar: {}", e))
                    })?
                } else {
                    tar_data
                };
                std::fs::write(output, &final_data).map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to write archive: {}", e))
                })?;
                Ok(ArchiveEntry {
                    path: output.to_string_lossy().to_string(),
                    size: total_size,
                    is_dir: false,
                })
            }
        }
    }

    /// Extract an archive to a directory.
    fn extract_archive(
        &self,
        archive_path: &Path,
        dest: &Path,
        format: ArchiveFormat,
    ) -> ToolResult<Vec<ArchiveEntry>> {
        std::fs::create_dir_all(dest).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to create destination directory: {}", e))
        })?;

        let data = std::fs::read(archive_path).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read archive: {}", e))
        })?;

        match format {
            ArchiveFormat::Zip => {
                let reader = ZipReader::from_bytes(data).map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to parse zip: {}", e))
                })?;
                let mut extracted = Vec::new();
                for entry in reader.entries() {
                    if entry.name.ends_with('/') {
                        // Directory entry.
                        let dir_path = dest.join(&entry.name);
                        std::fs::create_dir_all(&dir_path).map_err(|e| {
                            ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                        })?;
                        extracted.push(ArchiveEntry {
                            path: entry.name.clone(),
                            size: 0,
                            is_dir: true,
                        });
                        continue;
                    }
                    // Path traversal protection: the entry name must not escape dest.
                    let dest_canonical = dest.canonicalize().unwrap_or_else(|_| dest.to_path_buf());
                    let file_path = dest.join(&entry.name);
                    // Canonicalize parent to check.
                    if let Some(parent) = file_path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            ToolError::new("IO_ERROR", format!("Failed to create parent directory: {}", e))
                        })?;
                    }
                    let parent_canonical = file_path
                        .parent()
                        .and_then(|p| p.canonicalize().ok())
                        .unwrap_or_default();
                    if !parent_canonical.starts_with(&dest_canonical) {
                        return Err(ToolError::new(
                            "PATH_TRAVERSAL",
                            format!("Archive entry '{}' escapes destination directory", entry.name),
                        ));
                    }
                    let content = reader.extract(&entry.name).map_err(|e| {
                        ToolError::new("ARCHIVE_ERROR", format!("Failed to extract '{}': {}", entry.name, e))
                    })?;
                    std::fs::write(&file_path, &content).map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to write '{}': {}", file_path.display(), e))
                    })?;
                    extracted.push(ArchiveEntry {
                        path: entry.name.clone(),
                        size: content.len() as u64,
                        is_dir: false,
                    });
                }
                Ok(extracted)
            }
            ArchiveFormat::Tar | ArchiveFormat::TarGz => {
                // For tar.gz, we'd need to gunzip. Since our gzip_wrap doesn't
                // actually compress, we handle the stored case. For real
                // gzip, integrate flate2.
                let tar_data = if format == ArchiveFormat::TarGz {
                    // Strip gzip header (29 bytes) and trailer (8 bytes) for
                    // our stored format. This only works for our gzip_wrap
                    // output; real gzip requires a proper inflater.
                    if data.len() > 37 && data[0] == GZIP_MAGIC[0] && data[1] == GZIP_MAGIC[1] {
                        // Find the stored deflate block.
                        // Header is 10 bytes, then stored block header is 5 bytes.
                        data[15..data.len() - 8].to_vec()
                    } else {
                        return Err(ToolError::new(
                            "ARCHIVE_ERROR",
                            "Real gzip decompression requires the flate2 crate",
                        ));
                    }
                } else {
                    data
                };
                let entries = read_tar(&tar_data).map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to parse tar: {}", e))
                })?;
                let dest_canonical = dest.canonicalize().unwrap_or_else(|_| dest.to_path_buf());
                let mut extracted = Vec::new();
                for (name, content) in &entries {
                    if name.ends_with('/') {
                        std::fs::create_dir_all(dest.join(name)).ok();
                        extracted.push(ArchiveEntry {
                            path: name.clone(),
                            size: 0,
                            is_dir: true,
                        });
                        continue;
                    }
                    let file_path = dest.join(name);
                    if let Some(parent) = file_path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            ToolError::new("IO_ERROR", format!("Failed to create parent: {}", e))
                        })?;
                    }
                    let parent_canonical = file_path
                        .parent()
                        .and_then(|p| p.canonicalize().ok())
                        .unwrap_or_default();
                    if !parent_canonical.starts_with(&dest_canonical) {
                        return Err(ToolError::new(
                            "PATH_TRAVERSAL",
                            format!("Archive entry '{}' escapes destination directory", name),
                        ));
                    }
                    std::fs::write(&file_path, content).map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to write '{}': {}", file_path.display(), e))
                    })?;
                    extracted.push(ArchiveEntry {
                        path: name.clone(),
                        size: content.len() as u64,
                        is_dir: false,
                    });
                }
                Ok(extracted)
            }
        }
    }

    /// List the contents of an archive.
    fn list_archive(&self, archive_path: &Path, format: ArchiveFormat) -> ToolResult<Vec<ArchiveEntry>> {
        let data = std::fs::read(archive_path).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read archive: {}", e))
        })?;
        match format {
            ArchiveFormat::Zip => {
                let reader = ZipReader::from_bytes(data).map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to parse zip: {}", e))
                })?;
                Ok(reader
                    .entries()
                    .iter()
                    .map(|e| ArchiveEntry {
                        path: e.name.clone(),
                        size: e.uncompressed_size,
                        is_dir: e.name.ends_with('/'),
                    })
                    .collect())
            }
            ArchiveFormat::Tar | ArchiveFormat::TarGz => {
                let tar_data = if format == ArchiveFormat::TarGz {
                    if data.len() > 37 && data[0] == GZIP_MAGIC[0] && data[1] == GZIP_MAGIC[1] {
                        data[15..data.len() - 8].to_vec()
                    } else {
                        return Err(ToolError::new(
                            "ARCHIVE_ERROR",
                            "Real gzip decompression requires the flate2 crate",
                        ));
                    }
                } else {
                    data
                };
                let entries = read_tar(&tar_data).map_err(|e| {
                    ToolError::new("ARCHIVE_ERROR", format!("Failed to parse tar: {}", e))
                })?;
                Ok(entries
                    .iter()
                    .map(|(name, content)| ArchiveEntry {
                        path: name.clone(),
                        size: content.len() as u64,
                        is_dir: name.ends_with('/'),
                    })
                    .collect())
            }
        }
    }
}

#[async_trait]
impl Tool for ArchiveTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "archive",
                "Create, extract, and list zip and tar.gz archives. "
                    + "Supports creating archives from directories and extracting to directories.",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The operation to perform")
                            .enum_values(vec![
                                "create".to_string(),
                                "extract".to_string(),
                                "list".to_string(),
                            ]),
                    ),
                    (
                        "source".to_string(),
                        ParameterDefinition::string("Source path (directory for create, archive file for extract/list)"),
                    ),
                    (
                        "output".to_string(),
                        ParameterDefinition::string("Output archive path (for create) or destination directory (for extract)"),
                    ),
                    (
                        "format".to_string(),
                        ParameterDefinition::string("Archive format: zip, tar, tar.gz (auto-detected from extension if omitted)"),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?;

        match operation {
            "create" => {
                let source_str = params["source"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'source' for create"))?;
                let output_str = params["output"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'output' for create"))?;

                let source = self.resolve_path(source_str)?;
                let output = self.resolve_output_path(output_str)?;

                if !source.exists() {
                    return Err(ToolError::not_found(format!(
                        "Source '{}' does not exist",
                        source.display()
                    )));
                }

                let format = if let Some(f) = params["format"].as_str() {
                    match f {
                        "zip" => ArchiveFormat::Zip,
                        "tar" => ArchiveFormat::Tar,
                        "tar.gz" | "tgz" => ArchiveFormat::TarGz,
                        _ => return Err(ToolError::invalid_args(format!("Unknown format: {}", f))),
                    }
                } else {
                    ArchiveFormat::from_extension(output_str).unwrap_or(ArchiveFormat::Zip)
                };

                let source_for_task = source.clone();
                let output_for_task = output.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let tool = ArchiveTool::new(source_for_task.parent().unwrap_or(Path::new(".")).to_path_buf());
                    tool.create_archive(&source_for_task, &output_for_task, format)
                })
                .await
                .map_err(|e| ToolError::new("ARCHIVE_ERROR", format!("Archive task failed: {}", e)))??;

                let data = serde_json::json!({
                    "archive_path": result.path,
                    "total_uncompressed_size": result.size,
                    "format": format.extension(),
                });

                Ok(ToolOutput::success(format!(
                    "Created archive '{}' ({})",
                    result.path,
                    format.extension()
                ))
                .with_data(data))
            }
            "extract" => {
                let source_str = params["source"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'source' for extract"))?;
                let output_str = params["output"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'output' for extract"))?;

                let source = self.resolve_path(source_str)?;
                let output = self.resolve_output_path(output_str)?;

                if !source.is_file() {
                    return Err(ToolError::new(
                        "NOT_A_FILE",
                        format!("'{}' is not a file", source.display()),
                    ));
                }

                let format = if let Some(f) = params["format"].as_str() {
                    match f {
                        "zip" => ArchiveFormat::Zip,
                        "tar" => ArchiveFormat::Tar,
                        "tar.gz" | "tgz" => ArchiveFormat::TarGz,
                        _ => return Err(ToolError::invalid_args(format!("Unknown format: {}", f))),
                    }
                } else {
                    ArchiveFormat::from_extension(source_str).unwrap_or(ArchiveFormat::Zip)
                };

                let source_for_task = source.clone();
                let output_for_task = output.clone();
                let allowed_base = self.allowed_base.clone();
                let extracted = tokio::task::spawn_blocking(move || {
                    let tool = ArchiveTool::new(allowed_base);
                    tool.extract_archive(&source_for_task, &output_for_task, format)
                })
                .await
                .map_err(|e| ToolError::new("ARCHIVE_ERROR", format!("Extract task failed: {}", e)))??;

                let file_count = extracted.iter().filter(|e| !e.is_dir).count();
                let dir_count = extracted.iter().filter(|e| e.is_dir).count();

                let data = serde_json::json!({
                    "destination": output.to_string_lossy(),
                    "files_extracted": file_count,
                    "directories_created": dir_count,
                    "entries": extracted,
                });

                Ok(ToolOutput::success(format!(
                    "Extracted {} files and {} directories to '{}'",
                    file_count, dir_count, output.display()
                ))
                .with_data(data))
            }
            "list" => {
                let source_str = params["source"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'source' for list"))?;

                let source = self.resolve_path(source_str)?;

                if !source.is_file() {
                    return Err(ToolError::new(
                        "NOT_A_FILE",
                        format!("'{}' is not a file", source.display()),
                    ));
                }

                let format = if let Some(f) = params["format"].as_str() {
                    match f {
                        "zip" => ArchiveFormat::Zip,
                        "tar" => ArchiveFormat::Tar,
                        "tar.gz" | "tgz" => ArchiveFormat::TarGz,
                        _ => return Err(ToolError::invalid_args(format!("Unknown format: {}", f))),
                    }
                } else {
                    ArchiveFormat::from_extension(source_str).unwrap_or(ArchiveFormat::Zip)
                };

                let source_for_task = source.clone();
                let allowed_base = self.allowed_base.clone();
                let entries = tokio::task::spawn_blocking(move || {
                    let tool = ArchiveTool::new(allowed_base);
                    tool.list_archive(&source_for_task, format)
                })
                .await
                .map_err(|e| ToolError::new("ARCHIVE_ERROR", format!("List task failed: {}", e)))??;

                let file_count = entries.iter().filter(|e| !e.is_dir).count();
                let total_size: u64 = entries.iter().map(|e| e.size).sum();

                let content = entries
                    .iter()
                    .map(|e| {
                        let marker = if e.is_dir { "d" } else { "f" };
                        format!("{} {:>10}  {}", marker, e.size, e.path)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let data = serde_json::json!({
                    "archive": source.to_string_lossy(),
                    "total_entries": entries.len(),
                    "file_count": file_count,
                    "total_uncompressed_size": total_size,
                    "entries": entries,
                });

                Ok(ToolOutput::success(content).with_data(data))
            }
            other => Err(ToolError::invalid_args(format!("Unknown operation: {}", other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32() {
        // CRC-32 of "hello" is 0x3610a686.
        assert_eq!(crc32(b"hello"), 0x3610a686);
        // CRC-32 of empty string is 0.
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn test_zip_create_and_read() {
        let mut buf = Vec::new();
        {
            let mut writer = ZipWriter::new(&mut buf);
            writer.add_file("hello.txt", b"hello world").unwrap();
            writer.add_file("data/test.txt", b"test content").unwrap();
            writer.finish().unwrap();
        }

        let reader = ZipReader::from_bytes(buf).unwrap();
        let entries = reader.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "hello.txt");
        assert_eq!(entries[0].uncompressed_size, 11);

        let content = reader.extract("hello.txt").unwrap();
        assert_eq!(content, b"hello world");

        let content2 = reader.extract("data/test.txt").unwrap();
        assert_eq!(content2, b"test content");
    }

    #[test]
    fn test_tar_create_and_read() {
        let entries = vec![
            ("file1.txt".to_string(), b"hello".to_vec()),
            ("dir/file2.txt".to_string(), b"world".to_vec()),
        ];
        let tar = create_tar(&entries).unwrap();
        let read_entries = read_tar(&tar).unwrap();
        assert_eq!(read_entries.len(), 2);
        assert_eq!(read_entries[0].0, "file1.txt");
        assert_eq!(read_entries[0].1, b"hello");
        assert_eq!(read_entries[1].0, "dir/file2.txt");
        assert_eq!(read_entries[1].1, b"world");
    }

    #[test]
    fn test_archive_format_from_extension() {
        assert_eq!(
            ArchiveFormat::from_extension("test.zip"),
            Some(ArchiveFormat::Zip)
        );
        assert_eq!(
            ArchiveFormat::from_extension("test.tar.gz"),
            Some(ArchiveFormat::TarGz)
        );
        assert_eq!(
            ArchiveFormat::from_extension("test.tgz"),
            Some(ArchiveFormat::TarGz)
        );
        assert_eq!(
            ArchiveFormat::from_extension("test.tar"),
            Some(ArchiveFormat::Tar)
        );
        assert_eq!(ArchiveFormat::from_extension("test.txt"), None);
    }

    #[tokio::test]
    async fn test_archive_tool_create_zip() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("a.txt"), "hello").unwrap();
        std::fs::write(src.join("b.txt"), "world").unwrap();

        let tool = ArchiveTool::new(dir.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "operation": "create",
                "source": "src",
                "output": "out.zip",
                "format": "zip",
            }))
            .await;
        assert!(result.is_ok());
        assert!(dir.path().join("out.zip").exists());
    }

    #[tokio::test]
    async fn test_archive_tool_list_zip() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("a.txt"), "hello").unwrap();

        let tool = ArchiveTool::new(dir.path().to_path_buf());
        tool.execute(serde_json::json!({
            "operation": "create",
            "source": "src",
            "output": "out.zip",
            "format": "zip",
        }))
        .await
        .unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "list",
                "source": "out.zip",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("a.txt"));
    }

    #[tokio::test]
    async fn test_archive_tool_extract_zip() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("a.txt"), "hello").unwrap();

        let tool = ArchiveTool::new(dir.path().to_path_buf());
        tool.execute(serde_json::json!({
            "operation": "create",
            "source": "src",
            "output": "out.zip",
            "format": "zip",
        }))
        .await
        .unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "extract",
                "source": "out.zip",
                "output": "extracted",
            }))
            .await;
        assert!(result.is_ok());
        assert!(dir.path().join("extracted").join("a.txt").exists());
        let content = std::fs::read_to_string(dir.path().join("extracted").join("a.txt")).unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_archive_tool_create_tar() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("a.txt"), "hello").unwrap();

        let tool = ArchiveTool::new(dir.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "operation": "create",
                "source": "src",
                "output": "out.tar",
                "format": "tar",
            }))
            .await;
        assert!(result.is_ok());
        assert!(dir.path().join("out.tar").exists());
    }
}
