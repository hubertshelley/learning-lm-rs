use tokenizers::Tokenizer;

pub struct OutputGenerator {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl OutputGenerator {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true).unwrap()
    }

    pub fn next_token(&mut self, token: u32) -> Option<String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..]);
        if text.len() > prev_text.len() && text.chars().last()?.is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Some(text.1.to_string())
        } else {
            None
        }
    }
}
