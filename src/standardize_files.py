import os
import re
from collections import OrderedDict
import json

def clean_filename(old_name):
    """Convert filenames to standard format."""
    # First try to extract from existing standardized name
    parts = old_name.replace('.txt', '').split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}.txt"
    
    # CEO name mappings for non-standardized names
    ceo_mappings = {
        'Andrew Jassy-  (AMZN) CEO': ('andrew_jassy', 'amazon'),
        'Jane Fraser - Citigroup CEO': ('jane_fraser', 'citigroup'),
        'Mary Barra - General Motors Company CEO': ('mary_barra', 'gm'),
        'Karen Lynch - CVS CEO': ('karen_lynch', 'cvs'),
        'Carol Tome - UPS CEO': ('carol_tome', 'ups'),
        'Doug McMillon - Walmart CEO': ('doug_mcmillon', 'walmart'),
        'Gail Boudreaux - Elevance Health, I CEO': ('gail_boudreaux', 'elevance'),
        'Tricia Griffith - The Progressive C': ('tricia_griffith', 'progressive')
    }
    
    for old, (name, company) in ceo_mappings.items():
        if old in old_name:
            return f"{name}_{company}.txt"
    return old_name

def clean_sentence(sentence):
    """Clean and format a single sentence."""
    # Remove special characters but keep basic punctuation
    sentence = re.sub(r'[^\w\s\-.,!?\'"]', '', sentence)
    
    # Fix common issues
    sentence = re.sub(r'\s+', ' ', sentence)  # Multiple spaces
    sentence = re.sub(r'^\s*[,\-]\s*', '', sentence)  # Leading comma or dash
    sentence = re.sub(r'\s*[,\-]\s*$', '', sentence)  # Trailing comma or dash
    
    # Remove Q&A artifacts
    sentence = re.sub(r'^(?:Q|A)\s*[:.]?\s*', '', sentence)
    sentence = re.sub(r'(?i)(?:thank you|thanks)[^.]*$', '', sentence)
    sentence = re.sub(r'(?i)let me (?:have|ask) \w+ (?:to )?(?:give|answer|take).*$', '', sentence)
    
    # Clean up common transcription artifacts
    sentence = re.sub(r'(?i)^\s*(?:um|uh|so|well|yeah|okay|right)\s*,?\s*', '', sentence)
    sentence = re.sub(r'(?i)\s*(?:you know|i mean)\s*', ' ', sentence)
    
    # Fix common word issues
    sentence = re.sub(r'\bNd\b', 'And', sentence)
    sentence = re.sub(r'\bi\b', 'I', sentence)
    sentence = re.sub(r'\bwe re\b', "we're", sentence)
    sentence = re.sub(r'\bdont\b', "don't", sentence)
    sentence = re.sub(r'\bwont\b', "won't", sentence)
    sentence = re.sub(r'\bIm\b', "I'm", sentence)
    
    # Ensure proper capitalization
    sentence = sentence.strip()
    if sentence and sentence[0].isalpha():
        sentence = sentence[0].upper() + sentence[1:]
    
    # Ensure proper ending punctuation
    if sentence and not sentence[-1] in '.!?':
        sentence = sentence + '.'
    
    # Remove sentences that are just references to other speakers
    if re.match(r'^(?:Over to|Back to|Let me hand it to|I will let)\s+\w+\.?$', sentence):
        return ""
    
    return sentence

def split_into_sentences(text):
    """Split text into sentences more accurately."""
    # First clean up any weird spacing
    text = re.sub(r'\s+', ' ', text)
    
    # Common abbreviations to ignore
    abbrevs = {
        'mr.', 'mrs.', 'dr.', 'ms.', 'inc.', 'ltd.', 'corp.', 'co.',
        'u.s.', 'e.g.', 'i.e.', 'vs.', 'etc.', 'fig.', 'st.', 'sr.',
        'jr.', 'ph.d.', 'm.d.', 'b.a.', 'm.a.', 'q1.', 'q2.', 'q3.',
        'q4.', 'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 'aug.',
        'sep.', 'oct.', 'nov.', 'dec.'
    }
    
    # Split on sentence boundaries
    sentences = []
    current = []
    
    # First clean up some common issues
    text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n', text)  # Add newlines at sentence boundaries
    text = re.sub(r'(?<=\w)\.(?=\w)', '. ', text)  # Add space after period between words
    
    # Split into potential sentences
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if this line ends with an abbreviation
        words = line.lower().split()
        if words and words[-1] in abbrevs:
            current.append(line + ' ')
            continue
        
        # If it looks like a complete sentence, add it
        if re.search(r'[.!?]$', line):
            current.append(line)
            sentences.append(''.join(current).strip())
            current = []
        else:
            current.append(line + ' ')
    
    if current:
        sentences.append(''.join(current).strip())
    
    # Clean each sentence
    sentences = [clean_sentence(s) for s in sentences if s.strip()]
    
    # Remove very short or incomplete sentences
    sentences = [s for s in sentences if s and len(s.split()) >= 5]
    
    return sentences

def remove_duplicates(text):
    """Remove duplicate paragraphs and sentences while maintaining order."""
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Use OrderedDict to maintain order while removing duplicates
    unique_paragraphs = OrderedDict()
    
    for para in paragraphs:
        # Normalize the paragraph for comparison
        norm_para = re.sub(r'\s+', ' ', para.strip().lower())
        if norm_para and norm_para not in unique_paragraphs:
            unique_paragraphs[norm_para] = para.strip()
    
    return '\n\n'.join(unique_paragraphs.values())

def normalize_text_length(text, min_words=1000, max_words=2000):
    """Normalize text length while maintaining coherent paragraphs."""
    paragraphs = text.split('\n\n')
    total_words = sum(len(p.split()) for p in paragraphs)
    
    # If text is too short, return as is
    if total_words < min_words:
        return text, total_words
    
    # If text is too long, select most representative paragraphs
    if total_words > max_words:
        selected_paragraphs = []
        current_words = 0
        
        # Always include the first few paragraphs for context
        intro_paragraphs = paragraphs[:2]
        selected_paragraphs.extend(intro_paragraphs)
        current_words += sum(len(p.split()) for p in intro_paragraphs)
        
        # Select paragraphs from the middle
        middle_start = len(paragraphs) // 4
        middle_end = 3 * len(paragraphs) // 4
        middle_paragraphs = paragraphs[middle_start:middle_end]
        
        for para in middle_paragraphs:
            para_words = len(para.split())
            if current_words + para_words <= max_words - 200:  # Leave room for conclusion
                selected_paragraphs.append(para)
                current_words += para_words
        
        # Include some concluding paragraphs
        conclusion_paragraphs = paragraphs[-2:]
        selected_paragraphs.extend(conclusion_paragraphs)
        
        return '\n\n'.join(selected_paragraphs), current_words
    
    return text, total_words

def extract_metadata(text, filename):
    """Extract and return metadata about the speech."""
    # Try to determine context
    context = "unknown"
    if re.search(r'(?i)earnings.call|quarterly.results|q[1-4]', text):
        context = "earnings_call"
    elif re.search(r'(?i)interview|conversation.with', text):
        context = "interview"
    elif re.search(r'(?i)conference|summit|forum', text):
        context = "conference"
    
    # Try to extract date
    date_match = re.search(r'(?i)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}', text)
    date = date_match.group(0) if date_match else "unknown"
    
    # Count words and sentences
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    
    return {
        "filename": filename,
        "context": context,
        "date": date,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0
    }

def standardize_content(text):
    """Standardize the content format."""
    # Extract header information
    lines = text.split('\n')
    header = lines[0] if lines else ""
    
    # Extract name and company
    if '-' in header:
        name = header.split('-')[0].strip()
        company = header.split('-')[1].replace('CEO', '').strip()
    else:
        name = header.strip()
        company = ""
    
    # Remove header and any metadata from text
    text = '\n'.join(lines[1:])
    text = re.sub(r'^.*?(Q[1-4]|Quarter)\s+\d{4}.*?\n', '', text, flags=re.IGNORECASE|re.MULTILINE)
    text = re.sub(r'^.*?Earnings Call.*?\n', '', text, flags=re.IGNORECASE|re.MULTILINE)
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove speaker indicators and Q&A formatting
    text = re.sub(r'\b(?:Q:|A:|Speaker \d+:|\w+:)\s*', '', text)
    text = re.sub(r'(?i)(?:question|answer):\s*', '', text)
    text = re.sub(r'(?i)next question(?:\s+from\s+[^.]+)?\.?\s*', '', text)
    text = re.sub(r'(?i)(?:over to you|back to you|let me hand it to|i will let)\s+\w+[.]?\s*', '', text)
    
    # Split into sentences and clean each one
    sentences = split_into_sentences(text)
    
    # Group sentences into smaller paragraphs (max 3 sentences per paragraph)
    # This helps BERT process the text better
    paragraphs = []
    current_para = []
    word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # Start new paragraph if current one is too long
        if word_count + sentence_words > 100 or len(current_para) >= 3:
            if current_para:
                paragraphs.append(' '.join(current_para))
            current_para = []
            word_count = 0
        
        current_para.append(sentence)
        word_count += sentence_words
    
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # Join paragraphs with double newlines
    text = '\n\n'.join(paragraphs)
    
    # Remove duplicate content
    text = remove_duplicates(text)
    
    # Normalize text length
    text, word_count = normalize_text_length(text)
    
    # Format header
    header = f"{name} - {company} CEO" if company else f"{name} - CEO"
    text = f"{header}\n\n{text}"
    
    return text

def process_directory(directory_path):
    """Process all files in the directory."""
    processed_files = set()
    metadata_list = []
    
    for filename in os.listdir(directory_path):
        if not filename.endswith('.txt'):
            continue
            
        old_path = os.path.join(directory_path, filename)
        new_filename = clean_filename(filename)
        new_path = os.path.join(directory_path, new_filename)
        
        # Skip if already processed
        if filename in processed_files:
            continue
            
        # Read and clean content
        with open(old_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean and standardize content
        cleaned_content = standardize_content(content)
        
        # Extract metadata
        metadata = extract_metadata(cleaned_content, new_filename)
        metadata_list.append(metadata)
        
        # Write cleaned content to new file
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        # Remove old file if name changed
        if old_path != new_path and os.path.exists(old_path):
            os.remove(old_path)
        
        processed_files.add(new_filename)
        print(f"Processed: {filename} -> {new_filename} ({metadata['word_count']} words)")
    
    # Save metadata
    metadata_file = os.path.join(directory_path, "speech_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    return metadata_list

if __name__ == "__main__":
    ceo_directory = "data/speeches/ceos"
    print("Standardizing CEO speech files...")
    process_directory(ceo_directory)
    print("Done!") 