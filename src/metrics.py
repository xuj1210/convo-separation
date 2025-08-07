import pandas as pd
import jiwer
import re
import string
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from nltk.translate.meteor_score import meteor_score
import contractions
# import nltk

def normalize_text(text: str) -> str:
    """
    Normalizes a given string for accurate ASR metric calculation,
    notably handling slang and common transcription errors.
    """
    text = text.lower().replace('’', "'") # Standardize curly apostrophes
    text = contractions.fix(text)

    
    # punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # spaces
    text = re.sub(r'\s+', ' ', text).strip()

    print(text)
    
    return text

def parse_transcript_to_segments(transcript_path: str):
    """
    Parses transcript file into a list of segments.
    Specifically filters descriptor tags throughout the text leaving just words
    """
    segments = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        timestamp_match = re.match(r'\[(\d+\.\d+)\]', line)
        if timestamp_match:
            start_time = float(timestamp_match.group(1))
            i += 1
            if i < len(lines):
                speaker_text_line = lines[i].strip()
                speaker_match = re.search(r'<Speaker_(\d+)>', speaker_text_line)
                if speaker_match:
                    speaker_label = f'Speaker_{speaker_match.group(1)}'
                    current_text_lines = [re.sub(r'<Speaker_\d+>', '', speaker_text_line).strip()]
                    j = i + 1
                    while j < len(lines) and not re.match(r'\[(\d+\.\d+)\]', lines[j].strip()):
                        current_text_lines.append(lines[j].strip())
                        j += 1
                    if j < len(lines):
                        end_time = float(re.match(r'\[(\d+\.\d+)\]', lines[j].strip()).group(1))
                    else:
                        end_time = start_time + 10.0
                    full_text = ' '.join(current_text_lines)
                    clean_text = re.sub(r'<[^>]+>', '', full_text).strip()
                    clean_text = re.sub(r' +', ' ', clean_text)
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'speaker': speaker_label,
                        'text': clean_text
                    })
                    i = j
                    continue
        i += 1
    return segments

def load_from_csv(csv_path: str):
    """Loads a CSV file into a list of dictionaries."""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')

def calculate_base_metrics(reference_segments, hypothesis_segments, speaker_map):
    """
    Prepares lists for WER and other metrics and calculates DER.
    Returns:
        tuple: (SA-WER, DER, wer_references, wer_hypotheses, error_df)
    """
    wer_references = []
    wer_hypotheses = []
    total_duration = 0
    diarization_error_duration = 0
    
    ref_idx = 0
    hyp_idx = 0
    
    errors = []

    while ref_idx < len(reference_segments) and hyp_idx < len(hypothesis_segments):
        ref_seg = reference_segments[ref_idx]
        hyp_seg = hypothesis_segments[hyp_idx]
        
        overlap_start = max(ref_seg['start'], hyp_seg['start'])
        overlap_end = min(ref_seg['end'], hyp_seg['end'])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration > 0:
            mapped_speaker = speaker_map.get(hyp_seg['speaker'], hyp_seg['speaker'])
            
            if ref_seg['speaker'] != mapped_speaker:
                diarization_error_duration += overlap_duration
            else:
                normalized_ref = normalize_text(ref_seg['text'])
                normalized_hyp = normalize_text(hyp_seg['text'])
                
                if normalized_ref and normalized_hyp:
                    wer_references.append(normalized_ref)
                    wer_hypotheses.append(normalized_hyp)

                    # Get detailed word-level errors from the alignments
                    processed_words = jiwer.process_words(normalized_ref, normalized_hyp)
                    ref_words = processed_words.references[0]
                    hyp_words = processed_words.hypotheses[0]
                    
                    for alignment in processed_words.alignments[0]:
                        # Extract the error type and corresponding words
                        error_type = alignment.type
                        ref_word = None
                        hyp_word = None
                        
                        if error_type == 'substitute':
                            ref_words_list = ref_words[alignment.ref_start_idx : alignment.ref_end_idx]
                            hyp_words_list = hyp_words[alignment.hyp_start_idx : alignment.hyp_end_idx]
                            ref_word = ' '.join(ref_words_list)
                            hyp_word = ' '.join(hyp_words_list)
                            
                        elif error_type == 'insert':
                            hyp_words_list = hyp_words[alignment.hyp_start_idx : alignment.hyp_end_idx]
                            hyp_word = ' '.join(hyp_words_list)
                            
                        elif error_type == 'delete':
                            ref_words_list = ref_words[alignment.ref_start_idx : alignment.ref_end_idx]
                            ref_word = ' '.join(ref_words_list)
                        
                        if error_type != 'equal':
                            errors.append({
                                'error_type': error_type,
                                'ref_word': ref_word,
                                'hyp_word': hyp_word,
                                'start_time': ref_seg['start'],
                                'end_time': ref_seg['end'],
                                'speaker': ref_seg['speaker'],
                                'full_ref_text': ref_seg['text'],
                                'full_hyp_text': hyp_seg['text']
                            })

        if ref_seg['end'] <= hyp_seg['end']:
            ref_idx += 1
        if hyp_seg['end'] <= ref_seg['end']:
            hyp_idx += 1
            
        total_duration = max(total_duration, ref_seg['end'], hyp_seg['end'])

    sa_wer = jiwer.wer(reference=wer_references, hypothesis=wer_hypotheses)
    der = diarization_error_duration / total_duration if total_duration > 0 else 0.0

    error_df = pd.DataFrame(errors)
    
    return sa_wer, der, wer_references, wer_hypotheses, error_df

def calculate_bert_score(references, hypotheses):
    if not references or not hypotheses:
        return 0.0
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    return F1.mean().item()

def calculate_sentence_embedding_similarity(references, hypotheses):
    if not references or not hypotheses:
        return 0.0
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    hyp_embeddings = model.encode(hypotheses, convert_to_tensor=True)
    cosine_scores = util.cos_sim(ref_embeddings, hyp_embeddings)
    return cosine_scores.diag().mean().item()

def calculate_meteor_score(references, hypotheses):
    if not references or not hypotheses:
        return 0.0
    refs_tokenized = [[ref.split()] for ref in references]
    hyps_tokenized = [hyp.split() for hyp in hypotheses]
    
    scores = [meteor_score(refs_tokenized[i], hyps_tokenized[i]) for i in range(len(references))]
    return sum(scores) / len(scores)

if __name__ == "__main__":
    ground_truth_path = "data/USE_ASR003_sample/TRANSCRIPTION_SEGMENTED_TO_SENTENCES/F2308F2308/F2308F2308_USA_USA_001.txt"
    generated_csv_path = "output/sample.csv"

    reference_segments = parse_transcript_to_segments(ground_truth_path)

    hypothesis_segments = load_from_csv(generated_csv_path)

    speaker_mapping = {
        'SPEAKER_00': 'Speaker_2',
        'SPEAKER_01': 'Speaker_1'
    }

    sa_wer, der, wer_references, wer_hypotheses, error_df = calculate_base_metrics(
        reference_segments, hypothesis_segments, speaker_mapping
    )

    bert_f1 = calculate_bert_score(wer_references, wer_hypotheses)
    sentence_sim = calculate_sentence_embedding_similarity(wer_references, wer_hypotheses)
    meteor_s = calculate_meteor_score(wer_references, wer_hypotheses)

    error_df.to_csv('src/key-differences.csv', index=False)
    print("\nSuccessfully saved word-level errors to 'src/key-differences.csv'.")

    print("\n--- ASR and Diarization Metrics ---")
    print(f"Speaker-Attributed Word Error Rate (SA-WER): {sa_wer:.2%}")
    print(f"Diarization Error Rate (DER): {der:.2%}")

    print("\n--- Semantic Metrics (High is Better) ---")
    print(f"BERTScore (F1): {bert_f1:.4f}")
    print(f"Sentence Embedding Similarity: {sentence_sim:.4f}")
    print(f"METEOR Score: {meteor_s:.4f}")