"""
Curated Bangla Dataset Loader
Uses high-quality cleaned Bangla corpus from HuggingFace
"""
from datasets import load_dataset
from pathlib import Path
import re


def download_curated_bangla_corpus(output_path, target_size_mb=20):
    """
    Download curated, cleaned Bangla corpus from HuggingFace.
    
    Options (in order of preference):
    1. BanglaNLP - High-quality news corpus with quality filtering
    2. Raw Text Corpus - Cleaned and deduplicated
    3. csebuetnlp/IndicNLPSuite - Large-scale curated corpus
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading CURATED Bangla corpus...")
    print("Using: csebuetnlp/IndicNLPSuite (high-quality, deduplicated)")
    
    try:
        # Option 1: Try IndicNLP Suite (best quality)
        dataset = load_dataset(
            "ai4bharat/IndicNLPSuite", 
            "bn",  # Bengali
            split="train",
            streaming=True
        )
        source = "IndicNLPSuite (AI4Bharat)"
    except Exception as e:
        print(f"IndicNLP failed: {e}")
        print("Falling back to Wikipedia (cleaned)...")
        # Fallback: Use Wikipedia but with better filtering
        dataset = load_dataset(
            'wikimedia/wikipedia', 
            '20231101.bn', 
            split='train', 
            streaming=True
        )
        source = "Wikipedia (cleaned)"
    
    target_bytes = target_size_mb * 1024 * 1024
    current_size = 0
    texts = []
    
    print(f"Target size: {target_size_mb}MB")
    print("Downloading and cleaning...")
    
    for i, article in enumerate(dataset):
        # Extract text
        if 'text' in article:
            text = article['text']
        elif 'content' in article:
            text = article['content']
        else:
            continue
        
        # AGGRESSIVE CLEANING
        text = clean_bangla_text(text)
        
        if len(text) < 100:  # Skip very short texts
            continue
        
        texts.append(text)
        current_size += len(text.encode('utf-8'))
        
        if i % 100 == 0:
            print(f"  Downloaded: {current_size / 1024 / 1024:.2f}MB ({len(texts)} articles)")
        
        if current_size >= target_bytes:
            break
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\\n\\n'.join(texts))
    
    final_size = current_size / 1024 / 1024
    print(f"✓ Saved {final_size:.2f}MB to {output_path}")
    print(f"✓ Source: {source}")
    print(f"✓ Articles: {len(texts)}")
    
    return output_path


def clean_bangla_text(text):
    """
    Aggressive cleaning for Bangla text
    """
    # Remove URLs
    text = re.sub(r'http\\S+|www\\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\\S+@\\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove very short lines (likely noise)
    lines = text.split('\\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 20]
    
    # Split on Bangla sentence enders
    text = '\\n'.join(lines)
    text = re.sub(r'([।!?])\\s*', r'\\1\\n', text)
    
    # Remove lines with too many non-Bangla characters
    lines = text.split('\\n')
    cleaned_lines = []
    for line in lines:
        # Count Bangla characters (Unicode range: 0980-09FF)
        bangla_chars = len(re.findall(r'[\\u0980-\\u09FF]', line))
        total_chars = len(re.sub(r'\\s', '', line))
        
        if total_chars > 0 and bangla_chars / total_chars > 0.6:  # At least 60% Bangla
            cleaned_lines.append(line)
    
    text = '\\n'.join(cleaned_lines)
    
    # Remove duplicate lines
    lines = list(dict.fromkeys(text.split('\\n')))
    text = '\\n'.join(lines)
    
    return text.strip()


if __name__ == "__main__":
    # Test the downloader
    output = Path("data/bangla/bangla_curated.txt")
    download_curated_bangla_corpus(output, target_size_mb=20)
    
    # Print sample
    with open(output, 'r', encoding='utf-8') as f:
        sample = f.read(500)
    print("\\n" + "="*50)
    print("SAMPLE:")
    print("="*50)
    print(sample)
