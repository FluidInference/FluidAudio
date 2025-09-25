from kokoro import KPipeline, KModel
from kokoro.modules import AdaLayerNorm
from IPython.display import display, Audio
import soundfile as sf
import torch
import torch.nn as nn
import re
from misaki import en, espeak
import coremltools as ct
import torch.nn.functional as F
import math
import numpy as np

import numpy as np
pipeline = KPipeline(lang_code='a')

SAMPLE_RATE = 24000
text = '''hello we need to get to five seconds. hello we need to get to five seconds'''
text *= 2
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    duration_sec = len(audio) / SAMPLE_RATE
    print(f"{i}: {gs}, {ps}, duration={duration_sec:.3f} sec")
    display(Audio(data=audio, rate=SAMPLE_RATE))


from kokoro import KPipeline, KModel
from kokoro.modules import AdaLayerNorm
from IPython.display import display, Audio
import soundfile as sf
import torch
import torch.nn as nn
import re
from misaki import en, espeak
import coremltools as ct
import torch.nn.functional as F
import math
import numpy as np

def get_input_ids(
        pipeline,
        phonemes: str,
        context_length: int,
        speed: float = 1,
    ) -> torch.FloatTensor:
    
    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), phonemes)))
    assert len(input_ids)+2 <= context_length, (len(input_ids)+2, context_length)
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    return input_ids


def get_phonemes(
    pipeline,
    text: str,
    voice: str,
    speed: int = 1,
    split_pattern: str = r'\n+',
    lang_code = 'a'
) -> tuple[torch.FloatTensor, torch.LongTensor]:

    pack = pipeline.load_voice(voice)
    context_length = pipeline.model.bert.config.max_position_embeddings
    text = re.split(split_pattern, text.strip()) if split_pattern else [text]
    fallback = espeak.EspeakFallback(british=lang_code=='b')
    g2p = en.G2P(trf=False, british=lang_code=='b', fallback=fallback, unk='')
    # print (g2p)
            
    # Process each segment
    # print (text)
    for graphemes_index, graphemes in enumerate(text):
        if not graphemes.strip():  # Skip empty segments
            continue

        # print (graphemes_index, graphemes)
        # English processing (unchanged) 
        _, tokens = g2p(graphemes)
        # print (graphemes, tokens, len(tokens))
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            print (len(gs), len(ps), len(tks))
            if not ps:
                continue
            elif len(ps) > 510:
                ps = ps[:510]

            # TODO: Should return list
            input_ids = get_input_ids(pipeline, ps, context_length, speed)
            refs = pack[len(ps)-1]
            print(" ps: ", ps)
            return input_ids, refs

# Remove the trailing comma to make it a string instead of a tuple
# text = "The development of artificial intelligence has been one of the most transformative technological technological technological advances of the twenty."
text = '''hello we need to get to five seconds. hello we need to get to five seconds'''
text *= 2
 
input_ids, ref_s = get_phonemes(pipeline, text, "af_heart")

class TextEncoderPredictorFixed(nn.Module):
    """
    Fixed text_encoder that matches the exact Kokoro DurationEncoder implementation.
    Modified to avoid pack_padded_sequence issues in CoreML conversion.
    """
    
    def __init__(self, text_encoder):
        super().__init__()
        self.lstms = text_encoder.lstms  # Keep the original ModuleList structure
        self.d_model = text_encoder.d_model
        self.sty_dim = text_encoder.sty_dim
    
    def forward(self, x, style, text_lengths, m):
        """
        Forward pass matching DurationEncoder implementation.
        
        Args:
            x: Input tensor [batch, d_model, time]
            style: Style embedding [batch, style_dim]
            text_lengths: Lengths of sequences [batch]
            m: Mask tensor [batch, time]
        """
        masks = m
        # Permute: [batch, d_model, time] -> [time, batch, d_model]
        x = x.permute(2, 0, 1)
        
        # Expand style to match time dimension
        s = style.expand(x.shape[0], x.shape[1], -1)  # [time, batch, style_dim]
        
        # Concatenate with style
        x = torch.cat([x, s], axis=-1)  # [time, batch, d_model + style_dim]
        
        # Apply mask
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        
        # Transpose to batch first: [time, batch, features] -> [batch, time, features]
        x = x.transpose(0, 1)
        
        # Additional transpose: [batch, time, features] -> [batch, features, time]
        x = x.transpose(-1, -2)
        
        # Process through LSTM and AdaLayerNorm blocks
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                # AdaLayerNorm processing
                # Transpose for AdaLayerNorm: [batch, features, time] -> [batch, time, features]
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                
                # Re-concatenate style after normalization
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)  # [batch, features+style, time]
                
                # Apply mask
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                # LSTM processing
                # Transpose back: [batch, features, time] -> [batch, time, features]
                x = x.transpose(-1, -2)
                
                # For CoreML compatibility, avoid pack_padded_sequence
                # Instead, use regular LSTM with masking
                block.flatten_parameters()
                
                # Initialize hidden states explicitly for CoreML
                batch_size = x.shape[0]
                h0 = torch.zeros(2, batch_size, self.d_model // 2, dtype=x.dtype, device=x.device)
                c0 = torch.zeros(2, batch_size, self.d_model // 2, dtype=x.dtype, device=x.device)
                
                x, _ = block(x, (h0, c0))
                
                # Transpose back: [batch, time, features] -> [batch, features, time]
                x = x.transpose(-1, -2)
                
                # Pad if necessary to match mask dimensions
                if x.shape[-1] < m.shape[-1]:
                    x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device, dtype=x.dtype)
                    x_pad[:, :, :x.shape[-1]] = x
                    x = x_pad
        
        # Final transpose: [batch, features, time] -> [batch, time, features]
        return x.transpose(-1, -2)

class TextEncoderFixed(nn.Module):                                                                                                                       
     """                                                                                                                                                
     Fixed TextEncoder that avoids pack_padded_sequence and handles LSTM states explicitly.   
     """                                                                                                                                          

     def __init__(self, original_text_encoder):
         super().__init__()
         # Copy all components from original
         self.embedding = original_text_encoder.embedding
         self.cnn = original_text_encoder.cnn
         self.lstm = original_text_encoder.lstm

         # Get LSTM configuration
         self.hidden_size = self.lstm.hidden_size
         self.num_layers = self.lstm.num_layers
         self.bidirectional = self.lstm.bidirectional
         self.num_directions = 2 if self.bidirectional else 1
                                        
     def forward(self, x, input_lengths, m):
         """                               
         Forward pass with fixed LSTM handling for CoreML.
  
         Args: 
             x: Input tensor [batch, seq_len]
             input_lengths: Actual lengths of sequences [batch]
             m: Mask tensor [batch, seq_len]
            
         Returns:
             Output tensor [batch, channels, seq_len]
         """ 
         # Embedding  
         x = self.embedding(x)  # [B, T, emb] 
                                       
         # Transpose for CNN processing 
         x = x.transpose(1, 2)  # [B, emb, T]
                       
         # Prepare mask
         m = m.unsqueeze(1) 
         x.masked_fill_(m, 0.0) 
                      
         # CNN layers 
         for c in self.cnn: 
             x = c(x)
             x.masked_fill_(m, 0.0) 
                           
         # Transpose back for LSTM
         x = x.transpose(1, 2)  # [B, T, chn]
                                    
         # Initialize LSTM states explicitly
         batch_size = x.shape[0]                                                                                                              
         h0 = torch.zeros(                                                                                                                        
             self.num_directions * self.num_layers,                                                                                           
             batch_size,                                                                                                                              
             self.hidden_size,                                                                                                                     
             dtype=x.dtype,                                                                                                                         
             device=x.device                                                                                                                   
         )                                                                                                                                     
         c0 = torch.zeros(                                                                                                                      
             self.num_directions * self.num_layers,                                                                                                   
             batch_size,                                                                                                                           
             self.hidden_size,                                                                                                                     
             dtype=x.dtype,                                                                                                                       
             device=x.device                                                                                                                            
         )                                                                                                                                            
                                                                                                                                                      
         # Flatten parameters for efficiency                                                                                                         
         self.lstm.flatten_parameters()                                                                                                                
                                                                                                                                                 
         # Run LSTM without pack_padded_sequence                                                                                                      
         # The masking will handle the variable lengths                                                                    
         x, (hn, cn) = self.lstm(x, (h0, c0))                                                                                                           
                                                                                                                                            
         # Transpose for output                                                                                                                        
         x = x.transpose(-1, -2)  # [B, chn, T]                                                                                                      
                                                                                                                                                
         # Pad if necessary                                                                                                                               
         if x.shape[-1] < m.shape[-1]:                                                                                                                  
             x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device, dtype=x.dtype)                                                  
             x_pad[:, :, :x.shape[-1]] = x                                                                                                    
             x = x_pad                                                                                                                                 
                                                                                                                                                 
         # Final masking                                                                                                                        
         x.masked_fill_(m, 0.0)                                                                                                                   
                                                                                                                                                   
         return x     

import torch
import torch.nn as nn
import numpy as np

class SineGenDeterministic(nn.Module):
    def __init__(self, original_sine_gen):
        super().__init__()
        self.sine_amp = original_sine_gen.sine_amp
        self.harmonic_num = original_sine_gen.harmonic_num
        self.dim = original_sine_gen.dim
        self.sampling_rate = original_sine_gen.sampling_rate
        self.voiced_threshold = original_sine_gen.voiced_threshold
        
        # Phase stabilization parameters
        self.phase_reset_interval = 1000  # Reset phase accumulation every N frames
        self.use_double_precision = True  # Use double precision for phase math

    def forward(self, f0, random_phases):
        batch_size, seq_len = f0.shape[:2]
        
        # Use higher precision for phase calculations
        if self.use_double_precision:
            f0 = f0.double()
            random_phases = random_phases.double()
        
        # UV detection with smooth transition
        uv = torch.sigmoid((f0 - self.voiced_threshold) * 0.5)
        
        # Generate harmonics
        harmonic_nums = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
        fn = f0 * harmonic_nums.view(1, 1, -1)
        rad_values = (fn / self.sampling_rate)
        
        # Apply random phases only at the beginning
        rad_values[:, 0, :] = rad_values[:, 0, :] + random_phases.squeeze(1)
        
        # **FIXED: Use segmented phase accumulation to prevent drift**
        phase_accum = self.stable_phase_accumulation(rad_values, seq_len)
        
        # Generate sine waves
        sine_waves = torch.sin(phase_accum) * self.sine_amp * uv
        
        # Convert back to original precision
        if self.use_double_precision:
            sine_waves = sine_waves.float()
            
        return sine_waves

    def stable_phase_accumulation(self, rad_values, seq_len):
        """Stable phase accumulation with periodic resets"""
        batch_size, _, dim = rad_values.shape
        
        if seq_len <= self.phase_reset_interval:
            # Short sequence: use normal accumulation
            phase_accum = torch.cumsum(rad_values, dim=1)
        else:
            # Long sequence: use segmented accumulation
            num_segments = (seq_len + self.phase_reset_interval - 1) // self.phase_reset_interval
            phase_segments = []
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * self.phase_reset_interval
                end_idx = min((seg_idx + 1) * self.phase_reset_interval, seq_len)
                
                # Accumulate within segment
                segment_rad = rad_values[:, start_idx:end_idx, :]
                segment_accum = torch.cumsum(segment_rad, dim=1)
                
                # Add phase offset from previous segment (except first)
                if seg_idx > 0:
                    prev_segment_end_phase = phase_segments[seg_idx-1][:, -1:, :]
                    segment_accum = segment_accum + prev_segment_end_phase
                
                phase_segments.append(segment_accum)
            
            phase_accum = torch.cat(phase_segments, dim=1)
        
        # Wrap phase to [0, 2π] to maintain numerical stability
        phase_accum = (phase_accum - torch.floor(phase_accum)) * 2 * np.pi
        
        return phase_accum

class SourceModuleHnNSFDeterministic(nn.Module):
    """Deterministic source that preserves prosody"""
    def __init__(self, original_source):
        super().__init__()
        self.sine_amp = original_source.sine_amp
        self.l_sin_gen = SineGenDeterministic(original_source.l_sin_gen)
        self.l_linear = original_source.l_linear
        self.l_tanh = original_source.l_tanh
    
    def forward(self, x, random_phases):
        """Generate harmonics preserving F0 variations"""
        sine_wavs = self.l_sin_gen(x, random_phases)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        
        return sine_merge

class GeneratorDeterministic(nn.Module):
    def __init__(self, original_generator):
        super().__init__()
        # Keep all original components
        self.num_kernels = original_generator.num_kernels
        self.num_upsamples = original_generator.num_upsamples
        self.noise_convs = original_generator.noise_convs
        self.noise_res = original_generator.noise_res
        self.ups = original_generator.ups
        self.resblocks = original_generator.resblocks
        self.post_n_fft = original_generator.post_n_fft
        self.conv_post = original_generator.conv_post
        self.reflection_pad = original_generator.reflection_pad
        self.stft = original_generator.stft
        self.f0_upsamp = original_generator.f0_upsamp
        self.m_source = SourceModuleHnNSFDeterministic(original_generator.m_source)
    
    def forward(self, x, s, f0, random_phases):
        # Generate harmonics
        f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source = self.m_source(f0_up, random_phases)
        har_source = har_source.transpose(1, 2).squeeze(1)
        
        # STFT
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)

        # Precompute style statistics once per resblock to avoid redundant FCs
        resblock_style_stats = []
        for block in self.resblocks:
            if hasattr(block, "prepare_style"):
                resblock_style_stats.append(block.prepare_style(s))
            else:
                resblock_style_stats.append(None)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har) ####### 
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            # Dimension matching
            if x_source.shape[2] != x.shape[2]:
                if x_source.shape[2] < x.shape[2]:
                    x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
                else:
                    x_source = x_source[:, :, :x.shape[2]]

            x = x + x_source

            res_outputs = []
            base_idx = i * self.num_kernels
            for j in range(self.num_kernels):
                block = self.resblocks[base_idx + j]
                style_stats = resblock_style_stats[base_idx + j]
                if style_stats is not None:
                    res_outputs.append(block(x, style_stats=style_stats))
                else:
                    res_outputs.append(block(x, s))
            x = torch.stack(res_outputs, dim=0).mean(dim=0)
        
        # Final processing
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        
        # Standard processing
        x_mag = x[:,:self.post_n_fft // 2 + 1, :]
        x_mag = torch.clamp(x_mag, min=-10, max=10)
        spec = torch.exp(x_mag)
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        
        # Generate audio
        audio = self.stft.inverse(spec, phase)
        
        return audio

  
class KokoroCompleteCoreML(nn.Module):
    """
    Complete end-to-end Kokoro model for CoreML.
    Uses the model's own duration predictions (which are correct).
    """
    
    def __init__(self, model, bert, bert_encoder, predictor, device="cpu", samples_per_frame=600, max_frames=600):
        super().__init__()
        self.model = model
        self.bert = bert
        self.bert_encoder = bert_encoder
        self.predictor = predictor
        self.predictor_text_encoder = TextEncoderPredictorFixed(predictor.text_encoder)
        self.text_encoder = TextEncoderFixed(model.text_encoder)
        self.device = device
        self.samples_per_frame = samples_per_frame
        self.max_frames = max_frames  # 200 frames = 5 seconds at 40fps

        # Frontend components
        self.F0_conv = model.decoder.F0_conv
        self.N_conv = model.decoder.N_conv
        self.encode = model.decoder.encode
        self.asr_res = model.decoder.asr_res
        
        # Decoder components
        self.decode_blocks = model.decoder.decode
        
        # Generator component
        self.generator = GeneratorDeterministic(model.decoder.generator)

    def build_frame_indices(self, pred_dur: torch.Tensor, seq_len: int):
        """Return token indices per frame and a mask without materialising [L x T] alignment."""
        device = pred_dur.device
        batch_size = pred_dur.size(0)

        # Cumulative duration boundaries per token
        cumulative = torch.cumsum(pred_dur, dim=1)

        # Frame positions are 1-based to match cumulative comparison
        frame_positions = torch.arange(1, self.max_frames + 1, device=device).unsqueeze(0)
        frame_positions = frame_positions.expand(batch_size, -1)

        # Determine which token owns each frame by counting boundaries already passed
        token_indices = torch.sum(
            frame_positions.unsqueeze(-1) > cumulative.unsqueeze(1), dim=-1
        )
        token_indices = torch.clamp(token_indices, min=0, max=seq_len - 1).to(torch.long)

        # Build frame mask (0 for padded frames beyond total duration)
        total_frames = torch.clamp(cumulative[:, -1:], max=self.max_frames)
        frame_mask = (frame_positions <= total_frames).to(pred_dur.dtype)

        return token_indices, frame_mask
    
    def forward(self, input_ids, ref_s, random_phases, attention_mask, speed=1.0):
        batch_size, L = input_ids.shape
    
        # ---- DTYPE SAFETY ----
        # CoreML will feed int32; BERT embeddings expect torch.long
        input_ids_long = input_ids.to(torch.long)
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.int32)
        text_mask_bool = (attention_mask == 0)
        input_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    
        # Text encoding
        bert_output = self.bert(input_ids_long, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_output).transpose(-1, -2)
        style = ref_s[:, 128:]
    
        # Predictor
        d = self.predictor_text_encoder(d_en, style, input_lengths, text_mask_bool)
        lstm_layers = self.predictor.lstm.num_layers * (2 if self.predictor.lstm.bidirectional else 1)
        h0 = torch.zeros(lstm_layers, batch_size, self.predictor.lstm.hidden_size, dtype=d.dtype, device=d.device)
        c0 = torch.zeros(lstm_layers, batch_size, self.predictor.lstm.hidden_size, dtype=d.dtype, device=d.device)
        x, _ = self.predictor.lstm(d, (h0, c0))
    
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1)
    
        # Zero PADs
        valid = attention_mask.to(dtype=pred_dur.dtype)
        pred_dur = pred_dur * valid
    
        total_frames = pred_dur.sum(dim=1)
        audio_length_samples = (total_frames * self.samples_per_frame).to(torch.int32)
    
        # Gather indices (as you had)
        frame_indices, frame_mask = self.build_frame_indices(pred_dur, L)
    
        d_features = d.transpose(-1, -2)
        gather_idx = frame_indices.unsqueeze(1).expand(-1, d_features.size(1), -1)
        en = torch.gather(d_features, 2, gather_idx) * frame_mask.unsqueeze(1)
    
        F0_pred, N_pred = self.predictor.F0Ntrain(en, style)
        t_en = self.text_encoder(input_ids_long, input_lengths, text_mask_bool)
        gather_idx_text = frame_indices.unsqueeze(1).expand(-1, t_en.size(1), -1)
        asr = torch.gather(t_en, 2, gather_idx_text) * frame_mask.unsqueeze(1)
    
        ref_s_style = ref_s[:, :128]
        F0_processed = self.F0_conv(F0_pred.unsqueeze(1))
        N_processed = self.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0_processed, N_processed], dim=1)
        x_encoded = self.encode(x, ref_s_style)
        asr_res = self.asr_res(asr)
    
        x_current = x_encoded
        for decode_block in self.decode_blocks:
            x_input = torch.cat([x_current, asr_res, F0_processed, N_processed], dim=1)
            x_current = decode_block(x_input, ref_s_style)
    
        audio = self.generator(x_current, ref_s_style, F0_pred, random_phases)
        return audio, audio_length_samples, pred_dur

random_phases = torch.randn(1, 9)  # Adjust size as needed
example_inputs = (input_ids, ref_s, random_phases)

# Your existing setup
k_model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

# Your existing setup

# Prepare your example inputs
random_phases = torch.randn(1, 9)  # Adjust size as needed
attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
example_inputs = (input_ids, ref_s, random_phases, attention_mask)



def convert_to_single_coreml_model(original_decoder, pipeline, example_inputs):
    """Convert to a single end-to-end CoreML model"""
    
    # Extract example inputs
    input_ids, ref_s, random_phases, attention_mask = example_inputs
    
    print("Converting Complete End-to-End Kokoro Model...")
    
    # Create the complete model
    complete_model = KokoroCompleteCoreML(
        pipeline, 
        pipeline.bert, 
        pipeline.bert_encoder, 
        pipeline.predictor
    )
    complete_model.eval()
    
complete_model = KokoroCompleteCoreML(k_model, k_model.bert, k_model.bert_encoder, k_model.predictor)
complete_model.eval()


# Test with different speeds
with torch.no_grad():
    # Normal speed (should match pipeline output ~3.1s)
    audio1, length1, m = complete_model(input_ids, ref_s, random_phases, attention_mask, speed=1.0)
    print(f"Speed 1.0: {length1.item()/24000:.3f}s")
    
    # 2x faster (should be ~1.55s)  
    audio2, length2, s = complete_model(input_ids, ref_s, random_phases, attention_mask, speed=2.0)
    print(f"Speed 2.0: {length2.item()/24000:.3f}s")

with torch.no_grad():
    traced = torch.jit.trace(complete_model, (input_ids, ref_s, random_phases, attention_mask))

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="ref_s", shape=ref_s.shape, dtype=np.float32),
        ct.TensorType(name="random_phases", shape=random_phases.shape, dtype=np.float32),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="audio", dtype=np.float32),
        ct.TensorType(name="audio_length_samples", dtype=np.int32),
        ct.TensorType(name="pred_dur", dtype=np.float32),
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,          # <- make sure ANE is used

)
mlmodel.save("kokoro_24_5s.mlpackage")

import numpy as np
import coremltools as ct
import torch

SAMPLES_PER_FRAME = 600

# --- utilities ---
def to_np_i32(x): return x.detach().cpu().numpy().astype(np.int32)
def to_np_f32(x): return x.detach().cpu().numpy().astype(np.float32)

def get_output_names(mlmodel):
    spec = mlmodel.get_spec()
    names = [o.name for o in spec.description.output]
    return names

def classify_output_names(mlmodel):
    names = get_output_names(mlmodel)
    # try semantic guesses
    name_audio = next((n for n in names if "audio" in n.lower()), None)
    name_len   = next((n for n in names if "length" in n.lower() or "samples" in n.lower()), None)
    name_pred  = next((n for n in names if "pred" in n.lower() or "dur" in n.lower()), None)

    # if anything missing, fall back by shape/type after a dummy predict
    return name_audio, name_len, name_pred

def frames_from_pred_and_mask(pred_dur_np, mask_t):
    if pred_dur_np.ndim == 2:
        pred_dur_np = pred_dur_np[0]
    tok = int(mask_t.sum().item())
    return int(np.sum(pred_dur_np[:tok]))

def run_coreml_once(mlmodel, input_ids, ref_s, random_phases, mask):
    # discover names (and print once)
    name_audio, name_len, name_pred = classify_output_names(mlmodel)
    out = mlmodel.predict({
        "input_ids": to_np_i32(input_ids),
        "ref_s": to_np_f32(ref_s),
        "random_phases": to_np_f32(random_phases),
        "attention_mask": to_np_i32(mask),
    })

    # If any name is None (or not present), do a robust fallback by inspecting shapes/dtypes
    if (name_audio is None or name_audio not in out or
        name_len   is None or name_len   not in out or
        name_pred  is None or name_pred  not in out):
        # classify by dtype/shape
        audio_key = None; len_key = None; pred_key = None
        for k, v in out.items():
            arr = np.array(v)
            if arr.dtype.kind in "fc" and arr.ndim == 3:
                audio_key = k
            elif arr.dtype.kind in "iu" and arr.size in (1, arr.shape[0]):  # scalar or [B]
                len_key = k
            elif arr.dtype.kind in "fc" and arr.ndim in (1,2):
                pred_key = k
        name_audio = name_audio or audio_key
        name_len   = name_len   or len_key
        name_pred  = name_pred  or pred_key

    # final fetch
    audio_len = int(np.array(out[name_len]).ravel()[0])
    pred_dur  = np.array(out[name_pred])  # (L,) or (1,L)
    frames    = frames_from_pred_and_mask(pred_dur, mask)
    return audio_len, frames, pred_dur

@torch.no_grad()
def run_torch_once(model, input_ids, ref_s, random_phases, mask):
    audio, audio_len_t, pred_dur_t = model(input_ids, ref_s, random_phases, mask)
    audio_len = int(audio_len_t.view(-1)[0].item()) if torch.is_tensor(audio_len_t) else int(audio_len_t)
    pred_dur  = pred_dur_t.detach().cpu().numpy()
    if pred_dur.ndim == 1: pred_dur = pred_dur[None, :]
    frames = frames_from_pred_and_mask(pred_dur, mask)
    return audio_len, frames, pred_dur

def check_len_consistency(tag, frames, audio_len):
    expected = frames * SAMPLES_PER_FRAME
    ok = (expected == audio_len)
    print(f"[{tag}] frames={frames} expected={expected} audio_len={audio_len} match={ok}")
    return ok

import numpy as np
import coremltools as ct
import soundfile as sf
import torch
import os
import re
import time

SAMPLE_RATE = 24000
MAX_TOKENS = 153   # must match export

def pad_to_max(input_ids: torch.Tensor, max_tokens: int):
    L = input_ids.shape[1]
    if L < max_tokens:
        pad = torch.zeros(1, max_tokens - L, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids = input_ids[:, :max_tokens]
        L = max_tokens
    return input_ids, L

def build_mask(orig_len: int, max_tokens: int, device):
    mask = torch.zeros(1, max_tokens, dtype=torch.int32, device=device)
    mask[0, :orig_len] = 1
    return mask

def run_coreml_with_pt_inputs(mlpackage_path: str, text: str, voice: str, out_wav="cm_from_pt.wav"):
    # 1) Use the REAL tokenizer + style from your PT pipeline
    input_ids, ref_s = get_phonemes(pipeline, text, voice)  # <-- your helper
    input_ids = input_ids.to(dtype=torch.long)
    input_ids, true_len = pad_to_max(input_ids, MAX_TOKENS)
    mask = build_mask(true_len, MAX_TOKENS, input_ids.device)
    random_phases = torch.zeros(1, 9, dtype=torch.float32, device=input_ids.device)  # deterministic

    # 2) CoreML run
    mlmodel = ct.models.MLModel(mlpackage_path)
    return mlmodel

LOG_DIR = os.path.join(os.getcwd(), "coreml_trims")
os.makedirs(LOG_DIR, exist_ok=True)

model = run_coreml_with_pt_inputs("kokoro_24_5s.mlpackage", "in-groups. in-groups. in-groups.", "af_heart")
def run(text):
    print(text)
    input_ids, ref_s = get_phonemes(pipeline, text, "af_heart")  # <-- your helper
    input_ids = input_ids.to(dtype=torch.long)
    input_ids, true_len = pad_to_max(input_ids, MAX_TOKENS)
    mask = build_mask(true_len, MAX_TOKENS, input_ids.device)
    phase_dim = 9
    random_phases = torch.empty(1, phase_dim, device=input_ids.device,
    dtype=torch.float32)
    random_phases.uniform_(0.0, 2 * math.pi)

    # 2) CoreML run
    out = model.predict({
        "input_ids":      input_ids.cpu().numpy().astype(np.int32),
        "ref_s":          ref_s.cpu().numpy().astype(np.float32),
        "random_phases":  random_phases.cpu().numpy().astype(np.float32),
        "attention_mask": mask.cpu().numpy().astype(np.int32),
    })
    
    # 3) Auto-pick output names, trim, save
    audio_key = next(k for k,v in out.items() if np.array(v).ndim == 3)
    len_key   = next(k for k,v in out.items() if np.array(v).dtype.kind in "iu")
    audio = np.array(out[audio_key])[0, 0]
    alen  = int(np.array(out[len_key]).ravel()[0])
    # display(Audio(data=audio, rate=24000))
    pred_dur = np.array(out['pred_dur'])[0].astype(np.int32)
    pred_dur_t = torch.from_numpy(pred_dur).to(input_ids.device)

    guard_samples = SAMPLES_PER_FRAME * 3
    coreml_trim = audio[:alen]
    custom_trim = audio[:max(0, alen - guard_samples)]

    coreml_frames = alen // SAMPLES_PER_FRAME
    custom_frames = custom_trim.size // SAMPLES_PER_FRAME

    print(f"[CoreML] frames={coreml_frames} (≈ {coreml_trim.size / SAMPLE_RATE:.3f}s)")
    display(Audio(coreml_trim, rate=SAMPLE_RATE))

    print(f"[Custom] frames={custom_frames} (≈ {custom_trim.size / SAMPLE_RATE:.3f}s)")
    display(Audio(custom_trim, rate=SAMPLE_RATE))

    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "sample"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{timestamp}_{slug}"
    path_core = os.path.join(LOG_DIR, f"{base}_coreml.wav")
    path_guard = os.path.join(LOG_DIR, f"{base}_guard.wav")
    sf.write(path_core, coreml_trim, SAMPLE_RATE)
    sf.write(path_guard, custom_trim, SAMPLE_RATE)
    print(f"Saved trims to {path_core} and {path_guard}")
    

run("Hi") # no blip
run("He is") # no blip
run("She is") # no blip
run("She is running") # no blip
run("They are coming") # no blip

run("He is an expertt") # blip

run("He originally comes from USA.") # no blip

run("Dr. Wangs house is in USA.") # no blip


run("He originally comes from U.S.A.") # barely noticable blip

run("Dr. Wangs house is in U.S.A.") # no blip
