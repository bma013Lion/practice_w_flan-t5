from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class FlanT5Model:
    """Wrapper class for FLAN-T5 model."""
    
    # - this init method is the constructor method that runs with a 
    #   new FlanT5Model object is created. 
    # - takes in the model name, which defaults to the google/flan-t5-base 
    #   model, and device which defaults to the first available GPU if 
    #   available, else CPU.
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base", 
        device: Optional[str] = None
    ):
        # - Stores the model name as an instance variable
        self.model_name = model_name
        # - Stores the device as an instance variable
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
       
        # - Will store the tokenizer that processes text into a format 
        #   the model understands.
        #  - A tokenizer converts raw text into a formatted tokens  that 
        #    machine learning models can understand
        self.tokenizer = None
        
        # - Will store the actual FLAN-T5 model.
        self.model = None
        
        # - Calls the load_model method to load the tokenizer and model.
        self.load_model()
        
    def load_model(self) -> None:
       
        # AutoTokenizer is a class from the transformers library 
        # that loads a tokenizer for the specified model.   
        #
        # - The AutoTokenizer automatically selects the correct 
        #   tokenizer class based on the model name
        # - The tokenizer converts between text and the numerical
        #   tokens the model understands
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # AutoModelForSeq2SeqLM is a class from the transformers library 
        # that loads a sequence-to-sequence language model for the specified 
        # model.   
        #
        # - The AutoModelForSeq2SeqLM automatically selects the correct 
        #   model class based on the model name
        # - The model is loaded with the specified device and data type
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            # - The model name is used to load the model
            self.model_name,
            # - The device_map is used to specify the device to load the model on
            device_map="auto" if self.device == "cuda" else None,
            
            # - The model uses the specified torch_dtype data type for its parameters 
            #   and inputs
            #   - torch.float16 is used for half precision (16-bit floating point)
            #   - torch.float32 is used for single precision (32-bit floating point)
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            
        # - The model is moved to the specified device
        #   - If the device is "cuda", the model is moved to the GPU
        #   - If the device is "cpu", the model is moved to the CPU
        #   - This is where the actual computation will happen
        ).to(self.device)
        
    def generate(
        self,
        
        # - The prompt is the text that will be used as input 
        #   to the model
        prompt: str,
        
        # - The max_length is the maximum number of tokens that
        #   the model will generate
        max_length: int = 200,
        
        # - The temperature is a parameter that controls the
        #   randomness of the model's output
        temperature: float = 0.5,
        
        # - The top_p is a parameter that controls the diversity 
        #   of the model's output
        top_p: float = 0.9,
    ) -> str:
        """Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt.
            max_length: Maximum length of the generated text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            
        Returns:
            Generated text.
        """
        
        # - The tokenizer converts the input text into tokens that 
        #   the model can understand
        # - The return_tensors="pt" argument specifies that the output 
        #   should be in PyTorch format (as a tensor)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # - The model generates a response to the input text
        # - The with torch.no_grad(): temporarily disables gradient 
        #   calculation, which can save memory and speed up 
        #   inference
        with torch.no_grad():
            # - The model generates a response to the input text
           
            # - The do_sample parameter enables sampling, which can lead to 
            #   more diverse and creative outputs
            # - The top_p parameter controls the diversity of the output
            # - The num_return_sequences parameter specifies the number of 
            #   sequences to generate
            outputs = self.model.generate(
                **inputs,
                # - The max_length parameter specifies the maximum number of tokens 
                #   to generate
                max_length=max_length,
                # - The temperature parameter controls the randomness of the output
                temperature=temperature,
                # - Setting the do_sample parameter to True tells the model not to 
                #   just pick the most likely token each time, but to sample from the 
                #   probablilty distribution of possible next tokend. This adds 
                #   randomness to make the output more variable and natural. Otherwitse 
                #   every input to the model would yield the same output. corresponding 
                #   to the input.
                do_sample=True,
                # - The top_p parameter controls the diversity of the output by 
                #   using nucleus sampling which considers the smallest set of
                #   tokens whose cumulative probability exceeds the top_p set value.
                top_p=top_p,
                # - The num_return_sequences parameter specifies the number of 
                #   response sequences to generate and return 
                num_return_sequences=1,
                no_repeat_ngram_size=2,  # Prevents repeating n-grams
                repetition_penalty=1.2,  # Penalty for repetition
            )
        
        # - outputs[0]:
        #   - outputs contains the model's generated token IDs (numbers representing words/subwords)
        #   - [0] selects the first (and only) generated sequence since we set num_return_sequences=1
        # - self.tokenizer.decode():
        #   - Converts the sequence of token IDs back into human-readable text
        #   - This is the reverse of what the tokenizer did when we first processed the input
        # - skip_special_tokens=True:
        #   - Removes special tokens that were added during tokenization
        #   - Examples of special tokens:
        #     - [CLS], [SEP] (in BERT)
        #     - <s>, </s> (start/end of sequence in T5)
        #     - Padding tokens
        #   - Ensures the output is clean and readable
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        self.model_name = model_name
        self._load_model()
        
        