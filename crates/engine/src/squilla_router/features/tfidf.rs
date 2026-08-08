//! sklearn `TfidfVectorizer(analyzer="char_wb")` transform port.

use super::params::TfidfParams;
use std::collections::HashMap;

/// sklearn char_wb char n-grams of a text: lowercase; split on whitespace;
/// each token padded with a space at both ends; for each n in [lo,hi] emit
/// every char n-gram of length n of the padded token (including ones spanning
/// the padding). Returns the n-gram strings in occurrence order (duplicates ok).
pub fn char_wb_ngrams(text: &str, ngram_range: (usize, usize)) -> Vec<String> {
    let mut out = Vec::new();
    for token in text.to_lowercase().split_whitespace() {
        let padded = format!(" {token} ");
        let chars: Vec<char> = padded.chars().collect();
        for n in ngram_range.0..=ngram_range.1 {
            if chars.len() < n {
                continue;
            }
            for i in 0..=chars.len() - n {
                out.push(chars[i..i + n].iter().collect());
            }
        }
    }
    out
}

/// Sparse TF-IDF of a text against a fitted vocabulary: for each n-gram in
/// `char_wb_ngrams` that is in `params.vocabulary`, count tf; sublinear_tf =
/// 1 + ln(tf); weight = sublinear_tf * idf[col]; then L2 row-normalize across
/// the matched columns. Returns (col_id, weight) pairs for cols with weight != 0.
pub fn tfidf_transform(text: &str, params: &TfidfParams) -> Vec<(usize, f64)> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for ngram in char_wb_ngrams(text, (params.ngram_range[0], params.ngram_range[1])) {
        if let Some(&col) = params.vocabulary.get(&ngram) {
            *counts.entry(col).or_insert(0) += 1;
        }
    }
    let mut out: Vec<(usize, f64)> = Vec::new();
    let mut norm_sq = 0.0;
    for (&col, &tf) in &counts {
        let weight = (1.0 + (tf as f64).ln()) * params.idf[col];
        if weight != 0.0 {
            norm_sq += weight * weight;
            out.push((col, weight));
        }
    }
    if norm_sq == 0.0 {
        return Vec::new();
    }
    let norm = norm_sq.sqrt();
    for (_, weight) in &mut out {
        *weight /= norm;
    }
    out.sort_unstable_by_key(|&(col, _)| col);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_wb_pads_tokens_and_lowercases() {
        let grams = char_wb_ngrams("AbC", (2, 3));
        assert!(grams.contains(&" ab".to_string()));
        assert!(grams.contains(&"abc".to_string()));
        assert!(grams.contains(&"bc ".to_string()));
    }

    #[test]
    fn char_wb_cjk_chars() {
        assert_eq!(
            char_wb_ngrams("你好", (2, 2)),
            vec![" 你".to_string(), "你好".to_string(), "好 ".to_string()]
        );
    }

    #[test]
    fn tfidf_is_l2_normalized_over_vocab_cols() {
        let mut vocabulary = HashMap::new();
        vocabulary.insert("ab".to_string(), 0usize);
        vocabulary.insert("bc".to_string(), 1usize);
        let params = TfidfParams {
            ngram_range: [2, 2],
            sublinear_tf: true,
            max_features: 10,
            vocabulary,
            idf: vec![2.0, 3.0],
        };
        let sparse = tfidf_transform("ab bc", &params);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse[0].0, 0);
        assert_eq!(sparse[1].0, 1);
        let norm_sq: f64 = sparse.iter().map(|&(_, w)| w * w).sum();
        assert!((norm_sq - 1.0).abs() < 1e-9);
        assert!((sparse[0].1 * 1.5 - sparse[1].1).abs() < 1e-9);
    }

    #[test]
    fn tfidf_empty_when_no_vocab_match() {
        let params = TfidfParams {
            ngram_range: [2, 4],
            sublinear_tf: true,
            max_features: 10,
            vocabulary: HashMap::new(),
            idf: vec![],
        };
        assert!(tfidf_transform("abc", &params).is_empty());
    }
}
