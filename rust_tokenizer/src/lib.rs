use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct BPETokenizer {
    byte_to_id: [u64; 256],
    id_to_byte: [u8; 256],
    merges: Vec<(u64, u64)>,
    vocab_size: usize,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn new() -> Self {
        let mut byte_to_id = [0u64; 256];
        let mut id_to_byte = [0u8; 256];
        for b in 0..=255u16 {
            byte_to_id[b as usize] = b as u64;
            id_to_byte[b as usize] = b as u8;
        }
        BPETokenizer {
            byte_to_id,
            id_to_byte,
            merges: Vec::new(),
            vocab_size: 256,
        }
    }

    fn encode(&self, text: &str) -> Vec<u64> {
        text.bytes().map(|b| self.byte_to_id[b as usize]).collect()
    }

    fn decode(&self, ids: Vec<u64>) -> String {
        ids.iter().filter_map(|id| {
            if *id < 256 { Some(self.id_to_byte[*id as usize] as char) } else { None }
        }).collect()
    }

    fn vocab_size(&self) -> usize { self.vocab_size }
}

#[pymodule]
fn aria_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BPETokenizer>()?;
    Ok(())
}
