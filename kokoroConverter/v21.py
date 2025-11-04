# %%
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

# %%


# %% [markdown]
# # Original Kokoro pipeline

# %%
pipeline = KPipeline(lang_code='a')

text = '''
hello'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    # sf.write(f'{i}.wav', audio, 24000)

# %%
pipeline.model.eval()
# pipeline.model

# %%
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
            return input_ids, refs

# %% [markdown]
# # key

# %%
# Remove the trailing comma to make it a string instead of a tuple
text = "The development of artificial intelligence has been one of the most transformative technological advances of the twenty."
# text = "I can't believe we finally made it to the summit after climbing for twelve exhausting hours through wind and rain, but wow, this view of the endless mountain ranges stretching to the horizon makes every single difficult step completely worth the journey."
text = "I can't believe we finally made it to the summit after climbing for twelve exhausting hours through wind and rain, but wow, this view of the endless mountain ranges stret."

input_ids, ref_s = get_phonemes(pipeline, text, "af_heart")

# %%
ref_s.shape

# %% [markdown]
# # Bert Encoder conversion

# %%
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

# %%
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

# %% [markdown]
# # Mostly Generator classes
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

  
class SineGenDeterministic(nn.Module):
    def __init__(self, original_sine_gen):
        super().__init__()
        self.sine_amp = original_sine_gen.sine_amp
        self.harmonic_num = original_sine_gen.harmonic_num
        self.dim = original_sine_gen.dim
        self.sampling_rate = original_sine_gen.sampling_rate
        self.voiced_threshold = original_sine_gen.voiced_threshold

    
    def forward(self, f0, random_phases):
        batch_size, seq_len = f0.shape[:2]
        
        # Proper UV detection with smooth transition
        uv = torch.sigmoid((f0 - self.voiced_threshold) * 0.5)  # Gentler slope
        
        # Generate harmonics
        harmonic_nums = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
        fn = f0 * harmonic_nums.view(1, 1, -1)
        rad_values = (fn / self.sampling_rate)
        
        # Apply random phases at t=0
        rad_values[:, 0, :] = rad_values[:, 0, :] + random_phases.squeeze(1)
        
        # Use wrapped phase accumulation
        phase_accum = torch.cumsum(rad_values, dim=1)
        phase_wrapped = (phase_accum - torch.floor(phase_accum)) * 2 * np.pi
        
        # Generate sine waves with proper UV masking
        sine_waves = torch.sin(phase_wrapped) * self.sine_amp * uv
        
        return sine_waves

# %%


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
        
        # Create deterministic noise that varies with F0
        noise = torch.sin(x * 100) * self.sine_amp / 3
        
        return sine_merge

# %%
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
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
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
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, s)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, s)
            x = xs / self.num_kernels
        
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
        
        # Add tiny continuous signal to prevent complete silence
        # This prevents the discontinuities that cause clicks
        t = torch.arange(audio.shape[-1], device=audio.device, dtype=audio.dtype)
        # tiny_signal = torch.sin(t * 2 * np.pi * 50 / 24000) * 0.0001  # 50Hz at -80dB
        audio = audio 
        
        return audio

# %% [markdown]
# ## Combiners

# %%

class KokoroCompleteCoreML(nn.Module):
    """
    Complete end-to-end Kokoro model in a single CoreML package.
    Takes text tokens and reference style, outputs final audio.
    """
    
    def __init__(self, model, bert, bert_encoder, predictor, device="cpu", samples_per_frame=600):
        super().__init__()
        self.model = model
        self.bert = bert
        self.bert_encoder = bert_encoder
        self.predictor = predictor
        self.predictor_text_encoder = TextEncoderPredictorFixed(predictor.text_encoder)
        self.text_encoder = TextEncoderFixed(model.text_encoder)
        self.device = device
        self.samples_per_frame = samples_per_frame  # 24000 Hz / 40 fps = 600

        # Frontend components (PreGenerator)
        self.F0_conv = model.decoder.F0_conv
        self.N_conv = model.decoder.N_conv
        self.encode = model.decoder.encode
        self.asr_res = model.decoder.asr_res
        
        # Decoder components (all 4 blocks)
        self.decode_blocks = model.decoder.decode
        
        # Generator component
        self.generator = GeneratorDeterministic(model.decoder.generator)
    
    def create_alignment_matrix(self, pred_dur, device, max_frames):
        """
        CoreML-friendly alignment: no in-place assignment, uses broadcasting only.
        Output shape: [1, seq_len, F], where F == max_frames (fixed) or total frames (clamped).
        """
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
    
        # --- FIX: allow PADs to stay at 0 ---
        pred_dur = torch.round(pred_dur).clamp(min=0)  # keep >=0, don't force pads to 1
        pred_dur_int = pred_dur.to(torch.int32)
    
        # Cumulative frame endpoints and starts
        cum = torch.cumsum(pred_dur_int, dim=0)
        starts = torch.nn.functional.pad(cum[:-1], (1, 0), value=0)
    
        if max_frames is None:
            max_frames = int(cum[-1].item())
        max_frames = int(max_frames)
    
        frame_grid = torch.arange(max_frames, device=device, dtype=torch.int32).unsqueeze(0)
        starts = starts.to(torch.int32).unsqueeze(1)
        ends   = cum.to(torch.int32).unsqueeze(1)
    
        mask = (frame_grid >= starts) & (frame_grid < ends)
        total = torch.clamp(cum[-1], max=max_frames).to(torch.int32)
        valid_cols = (frame_grid < total.unsqueeze(0))
        mask = mask & valid_cols

        return mask.to(torch.float32).unsqueeze(0)  # [1, seq_len, F]

        
    def forward(self, input_ids, ref_s, random_phases, attention_mask, speed: torch.Tensor, min_frames: torch.Tensor, target_seconds: torch.Tensor, dur_override: torch.Tensor):
        # allow dynamic speed control from CoreML input
        # expected shape: [1] or scalar; clamp to reasonable range
        if torch.is_tensor(speed):
            if speed.dim() > 0:
                speed = speed.view(-1)[0]
        speed = torch.clamp(speed, min=torch.tensor(0.25, dtype=torch.float32, device=input_ids.device), max=torch.tensor(4.0, dtype=torch.float32, device=input_ids.device))
        batch_size, L = input_ids.shape
    
        # --- Build mask (1=valid, 0=pad) ---
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.int32)  # [B, L]
        text_mask_bool = (attention_mask == 0)  # [B, L] True where PAD
        input_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)  # [B]
    
        # --- Encode text ---
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_output).transpose(-1, -2)
        style = ref_s[:, 128:]
    
        # --- Predictor ---
        d = self.predictor_text_encoder(d_en, style, input_lengths, text_mask_bool)
        lstm_layers = self.predictor.lstm.num_layers * (2 if self.predictor.lstm.bidirectional else 1)
        h0 = torch.zeros(lstm_layers, batch_size, self.predictor.lstm.hidden_size, dtype=d.dtype, device=d.device)
        c0 = torch.zeros(lstm_layers, batch_size, self.predictor.lstm.hidden_size, dtype=d.dtype, device=d.device)
        x, _ = self.predictor.lstm(d, (h0, c0))
        # Duration prediction (pre-round, float)
        duration_logits = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration_logits).sum(axis=-1) / speed  # [B, L]

        # Valid mask (1 = token, 0 = pad)
        valid = attention_mask.to(dtype=duration.dtype)                 # [B, L]
        raw = duration * valid

        # Min frames clamp + total-preserving rescale
        # Extract scalar controls
        if torch.is_tensor(min_frames):
            if min_frames.dim() > 0:
                min_frames = min_frames.view(-1)[0]
        min_frames = torch.clamp(min_frames, min=torch.tensor(0.0, dtype=raw.dtype, device=raw.device))
        eps = torch.tensor(1e-6, dtype=raw.dtype, device=raw.device)

        raw_min = torch.maximum(raw, min_frames) * valid
        total_raw = raw.sum(dim=1, keepdim=True) + eps
        total_min = raw_min.sum(dim=1, keepdim=True) + eps
        scale = total_raw / total_min
        scaled = raw_min * scale

        # Optional target seconds alignment
        if torch.is_tensor(target_seconds):
            ts = target_seconds.view(-1)[0] if target_seconds.dim() > 0 else target_seconds
        else:
            ts = torch.tensor(0.0, dtype=scaled.dtype, device=scaled.device)
        fps = torch.tensor(24000.0 / float(self.samples_per_frame), dtype=scaled.dtype, device=scaled.device)
        frames_target = torch.round(torch.clamp(ts, min=torch.tensor(0.0, dtype=scaled.dtype, device=scaled.device)) * fps)
        use_target = (frames_target > 0)
        total_scaled = scaled.sum(dim=1, keepdim=True) + eps
        scale2 = torch.where(use_target.view(-1,1), frames_target.view(-1,1) / total_scaled, torch.ones_like(total_scaled))
        scaled = scaled * scale2

        pred_dur = torch.round(scaled).clamp(min=1) * valid

        # Optional duration override (expects [B, L] float, 0 means unused)
        if dur_override is not None:
            if dur_override.dtype != pred_dur.dtype:
                dur_override = dur_override.to(dtype=pred_dur.dtype)
            # Ensure shape and mask
            dur_override = torch.round(torch.clamp(dur_override, min=torch.tensor(0.0, dtype=pred_dur.dtype, device=pred_dur.device))) * valid
            use_ovr = (dur_override.sum(dim=1, keepdim=True) > 0).to(pred_dur.dtype)
            pred_dur = pred_dur * (1.0 - use_ovr) + dur_override * use_ovr
        
        # Total frames & audio length (NO .item(), stays tensorized)
        total_frames = pred_dur.sum(dim=1)                           # [B], float
        audio_length_samples = (total_frames * self.samples_per_frame).to(torch.int32)  # [B]

    
        # --- Alignment (B=1 path using dynamic frames for alignment grid) ---
        # If your create_alignment_matrix expects 1D durations, pass pred_dur[0]
        # Using a Python int here only affects alignment grid size; audio_length_samples remains correct & trace-safe.
        total_frames_int = int(torch.clamp(total_frames[0], min=1).item())
        pred_aln_trg = self.create_alignment_matrix(pred_dur[0], device=self.device, max_frames=total_frames_int)  # [1, L, F]
        en = d.transpose(-1, -2) @ pred_aln_trg                                # (B=1) → (1, ch, F)
    
        F0_pred, N_pred = self.predictor.F0Ntrain(en, style)                   # (1, F) each
        t_en = self.text_encoder(input_ids, input_lengths, text_mask_bool)     # (B, C, L)
        asr = t_en @ pred_aln_trg                                              # (1, C, F)
    
        # --- Pre-generator ---
        ref_s_style = ref_s[:, :128]
        F0_processed = self.F0_conv(F0_pred.unsqueeze(1))                      # (1, c, F)
        N_processed = self.N_conv(N_pred.unsqueeze(1))                         # (1, c, F)
        x = torch.cat([asr, F0_processed, N_processed], dim=1)                 # (1, Csum, F)
        x_encoded = self.encode(x, ref_s_style)
        asr_res = self.asr_res(asr)
    
        # --- Decoder ---
        x_current = x_encoded
        for decode_block in self.decode_blocks:
            x_input = torch.cat([x_current, asr_res, F0_processed, N_processed], dim=1)
            x_current = decode_block(x_input, ref_s_style)
    
        audio = self.generator(x_current, ref_s_style, F0_pred, random_phases) # (1, 1, T_fixed)
    
        # Return batched shapes consistently:
        # audio_length_samples: (B,) int32
        # pred_dur: (B, L) float32
        return audio, audio_length_samples, pred_dur


# %% [markdown]
# # converting

# %%
random_phases = torch.randn(1, 9)  # Adjust size as needed
example_inputs = (input_ids, ref_s, random_phases)

# Your existing setup
k_model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

# Your existing setup

# Prepare your example inputs
random_phases = torch.randn(1, 9)  # Adjust size as needed
example_attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
example_inputs = (input_ids, ref_s, random_phases, example_attention_mask)


def convert_to_single_coreml_model(original_decoder, pipeline, example_inputs):
    """Convert to a single end-to-end CoreML model"""
    
    # Extract example inputs
    input_ids, ref_s, random_phases, example_attention_mask = example_inputs
    
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

example_speed = torch.tensor([1.0], dtype=torch.float32)
example_min_frames = torch.tensor([2.0], dtype=torch.float32)
example_target_seconds = torch.tensor([0.0], dtype=torch.float32)
example_dur_override = torch.zeros_like(example_attention_mask, dtype=torch.float32)
with torch.no_grad():
    traced = torch.jit.trace(complete_model, (input_ids, ref_s, random_phases, example_attention_mask, example_speed, example_min_frames, example_target_seconds, example_dur_override))

token_dim = ct.RangeDim(2, int(example_attention_mask.shape[1]))
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, token_dim), dtype=np.int32),
        ct.TensorType(name="ref_s", shape=ref_s.shape, dtype=np.float32),
        ct.TensorType(name="random_phases", shape=random_phases.shape, dtype=np.float32),
        ct.TensorType(name="attention_mask", shape=(1, token_dim), dtype=np.int32),
        ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
        ct.TensorType(name="min_frames", shape=(1,), dtype=np.float32),
        ct.TensorType(name="target_seconds", shape=(1,), dtype=np.float32),
        ct.TensorType(name="dur_override", shape=(1, token_dim), dtype=np.float32),
    ],
    outputs=[
        ct.TensorType(name="audio", dtype=np.float32),
        ct.TensorType(name="audio_length_samples", dtype=np.int32),
        ct.TensorType(name="pred_dur", dtype=np.float32),
        
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.iOS17
)
mlmodel.save("kokoro_completev21.mlpackage")

# Early exit to avoid running demo/benchmark blocks when invoked as a script
import sys as _sys
print("[ok] saved kokoro_completev21.mlpackage — exiting early to skip demos.")
_sys.exit(0)


# %% [markdown]
# # inference

# %%
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

def run_coreml_once(mlmodel, input_ids, ref_s, random_phases, mask, speed: float = 1.0, min_frames: float = 2.0, target_seconds: float = 0.0, dur_override: np.ndarray | None = None):
    # discover names (and print once)
    name_audio, name_len, name_pred = classify_output_names(mlmodel)
    if dur_override is None:
        dur_override = to_np_f32(mask) * 0.0
    out = mlmodel.predict({
        "input_ids": to_np_i32(input_ids),
        "ref_s": to_np_f32(ref_s),
        "random_phases": to_np_f32(random_phases),
        "attention_mask": to_np_i32(mask),
        "speed": np.array([speed], dtype=np.float32),
        "min_frames": np.array([min_frames], dtype=np.float32),
        "target_seconds": np.array([target_seconds], dtype=np.float32),
        "dur_override": dur_override.astype(np.float32),
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


# %%
import numpy as np
import coremltools as ct
import soundfile as sf
import torch

SAMPLE_RATE = 24000
MAX_TOKENS = 168   # must match export

def pad_to_max(input_ids: torch.Tensor, max_tokens: int):
    # Preserve true length BEFORE padding/clamping
    true_len = min(int(input_ids.shape[1]), int(max_tokens))
    ids = input_ids[:, :true_len]
    if true_len < max_tokens:
        pad = torch.zeros(1, max_tokens - true_len, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([ids, pad], dim=1)
    else:
        # already clamped to max_tokens
        pass
    return ids, true_len

def build_mask(orig_len: int, max_tokens: int, device):
    mask = torch.zeros(1, max_tokens, dtype=torch.int32, device=device)
    mask[0, :orig_len] = 1
    return mask

def run_coreml_with_pt_inputs(mlpackage_path: str, text: str, voice: str, out_wav="cm_from_pt.wav", speed: float = 1.0, min_frames: float = 2.0, target_seconds: float = 0.0, dur_override: np.ndarray | None = None):
    # 1) Use the REAL tokenizer + style from your PT pipeline
    input_ids, ref_s = get_phonemes(pipeline, text, voice)  # <-- your helper
    input_ids = input_ids.to(dtype=torch.long)
    input_ids, true_len = pad_to_max(input_ids, MAX_TOKENS)
    mask = build_mask(true_len, MAX_TOKENS, input_ids.device)
    random_phases = torch.zeros(1, 9, dtype=torch.float32, device=input_ids.device)  # deterministic

    # 2) CoreML run
    mlmodel = ct.models.MLModel(mlpackage_path)

    import threading, psutil, os, time

    def monitor_mem(interval=0.1):
        proc = psutil.Process(os.getpid())
        while not stop.is_set():
            rss = proc.memory_info().rss / 1024**2  # MB
            print(f"[mem] {rss:.1f} MB")
            time.sleep(interval)
    
    stop = threading.Event()
    t = threading.Thread(target=monitor_mem, daemon=True)
    t.start()


    if dur_override is None:
        dur_override = np.zeros((1, true_len), dtype=np.float32)
    out = mlmodel.predict({
        "input_ids":      input_ids.cpu().numpy().astype(np.int32),
        "ref_s":          ref_s.cpu().numpy().astype(np.float32),
        "random_phases":  random_phases.cpu().numpy().astype(np.float32),
        "attention_mask": mask.cpu().numpy().astype(np.int32),
        "speed":          np.array([speed], dtype=np.float32),
        "min_frames":     np.array([min_frames], dtype=np.float32),
        "target_seconds": np.array([target_seconds], dtype=np.float32),
        "dur_override":   dur_override.astype(np.float32),
    })
        
    stop.set()
    t.join()

    # 3) Auto-pick output names, trim, save
    audio_key = next(k for k,v in out.items() if np.array(v).ndim == 3)
    len_key   = next(k for k,v in out.items() if np.array(v).dtype.kind in "iu")
    audio = np.array(out[audio_key])[0, 0]
    alen  = int(np.array(out[len_key]).ravel()[0])
    audio = audio[:alen]
    display(Audio(data=audio, rate=24000))

    sf.write(out_wav, audio, SAMPLE_RATE)
    print(f"[ok] wrote {out_wav} | {alen/SAMPLE_RATE:.2f}s")

# Example


# %%
text =  "NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and"
text = "I can't believe we finally made it to the summit after climbing for twelve exhausting hours through wind and rain, but wow, this view of the endless mountain ranges stretching to the horizon makes every single difficult step completely worth the journey."


# %%
import os
if os.path.exists("kokoro_21_10s_ANE.mlpackage"):
    run_coreml_with_pt_inputs("kokoro_21_10s_ANE.mlpackage", text+".", "af_heart")
else:
    print("[skip] kokoro_21_10s_ANE.mlpackage not found; skipping demo run.")


# %%
import coremltools as ct
import coremltools.optimize as cto
import os as _os

if _os.path.exists("kokoro_21_15s.mlpackage"):
    mlmodel = ct.models.MLModel("kokoro_21_15s.mlpackage")  # single-function is safest

    # Build the INT8 weight quantization config
    op_cfg = cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",    # or "linear" (with zero-point)
        weight_threshold=512        # only layers with >= this many weights get quantized
        # granularity / per_channel etc. are inferred; tune via per-op configs if needed
    )
    cfg = cto.coreml.OptimizationConfig(global_config=op_cfg)

    # Quantize weights -> INT8 (W8)
    w8_model = cto.coreml.linear_quantize_weights(mlmodel, config=cfg)
    w8_model.save("kokoro_21_15s_int8.mlpackage")
else:
    print("[skip] kokoro_21_15s.mlpackage not found; skipping quantization demo.")


# %%
import coremltools as ct
print(ct.__version__)


# %%
# 15 seconds
conjunctions = [ 
"NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and",
"Neoliberal economics emphasizes free markets minimal government intervention privatization and", 
"deregulation it promotes individual responsibility competition and ",
"She likes tea, and",
"We could stay home, or",
"He tried his best, yet",
"They promised to come, so",
"I was tired, for",
"Call me when",
"Don’t go unless",
"It was both",
"She is either",
"I’ll go whether",
"It was late; however",
"The data were messy; therefore",
"We liked the plan; nevertheless"]

# %%
for t in conjunctions:
    print(t)
    run_coreml_with_pt_inputs("kokoro_completev21.mlpackage", t + ".", "af_heart")


# %%
