//! Golden outputs for FluidAudio's Swift port of the Phonon English text normalizer.
//!
//! Reads a JSON array of input strings and writes, for each, the real `normalize_text` output
//! (English, default rules), the `prepare_text_prompt` result, and the `split_into_best_sentences`
//! chunks of the normalized text under a deterministic toy tokenizer (so the chunking algorithm,
//! not a vocabulary, is what gets compared).
//!
//! Build inside the ptts workspace (add `[[example]] name = "phonon_normalize_golden"` to
//! `ptts/Cargo.toml`), then:
//!
//!     cargo run --release -p ptts --example phonon_normalize_golden -- corpus.json golden.json

use std::cell::RefCell;
use std::collections::HashMap;

use ptts::preprocess::{Normalize, Rules};
use ptts::tts_model::{prepare_text_prompt, split_into_best_sentences};

/// Pieces: a maximal run of alphanumeric chars, or any other single non-space char, each taking
/// a single preceding space with it; a space that precedes another space or ends the text is a
/// piece of its own. Ids are handed out in order of first appearance. Decoding concatenates the
/// pieces and drops one leading space, as a SentencePiece decoder does.
#[derive(Default)]
struct ToyTokenizer {
    ids: RefCell<HashMap<String, u32>>,
    pieces: RefCell<Vec<String>>,
}

impl ToyTokenizer {
    fn id(&self, piece: String) -> u32 {
        let mut ids = self.ids.borrow_mut();
        if let Some(&id) = ids.get(&piece) {
            return id;
        }
        let mut pieces = self.pieces.borrow_mut();
        let id = pieces.len() as u32;
        pieces.push(piece.clone());
        ids.insert(piece, id);
        id
    }
}

impl ptts::Tokenizer for ToyTokenizer {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = vec![];
        let mut i = 0;
        while i < chars.len() {
            let mut piece = String::new();
            if chars[i] == ' ' {
                piece.push(' ');
                i += 1;
                if i >= chars.len() || chars[i] == ' ' {
                    out.push(self.id(piece));
                    continue;
                }
            }
            if chars[i].is_alphanumeric() {
                while i < chars.len() && chars[i].is_alphanumeric() {
                    piece.push(chars[i]);
                    i += 1;
                }
            } else {
                piece.push(chars[i]);
                i += 1;
            }
            out.push(self.id(piece));
        }
        Ok(out)
    }

    fn decode(&self, tokens: &[u32]) -> xn::Result<String> {
        let pieces = self.pieces.borrow();
        let text: String = tokens.iter().map(|&t| pieces[t as usize].as_str()).collect();
        Ok(text.strip_prefix(' ').map(str::to_string).unwrap_or(text))
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: {} <corpus.json> <golden.json>", args[0]);
    let inputs: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let normalize = "en".parse::<Normalize>()?.with_rules("default".parse::<Rules>()?);
    let mut cases = vec![];
    for input in inputs {
        let output = normalize.apply(&input).into_owned();
        let (prompt, frames) = prepare_text_prompt(&input);
        let chunks50 = split_into_best_sentences(&ToyTokenizer::default(), &output, Some(50))?;
        let chunks12 = split_into_best_sentences(&ToyTokenizer::default(), &output, Some(12))?;
        cases.push(serde_json::json!({
            "input": input,
            "output": output,
            "prompt": prompt,
            "framesAfterEos": frames,
            "chunks50": chunks50,
            "chunks12": chunks12,
        }));
    }
    let json = serde_json::to_string_pretty(&cases)?;
    std::fs::write(&args[2], json + "\n")?;
    eprintln!("wrote {} cases to {}", cases.len(), args[2]);
    Ok(())
}
