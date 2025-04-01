"""
CEO Transcript Preprocessing Script

This script preprocesses CEO earnings call transcripts to extract only the CEO's speech
for more accurate personality analysis. It handles various transcript formats and
filters out other speakers (analysts, moderators, other executives).

Usage:
    python preprocess_ceo_transcripts.py --input_dir <input_directory> --output_dir <output_directory>

Args:
    --input_dir: Directory containing the raw transcript files
    --output_dir: Directory to save the preprocessed CEO-only transcripts
    --ceo_name_file: Optional CSV file mapping transcript files to CEO names
"""

import os
import re
import argparse
import csv
from typing import Dict, List, Tuple, Set


class TranscriptPreprocessor:
    def __init__(self):
        # Common speaker indicators
        self.moderator_indicators = [
            "operator", "moderator", "host", "coordinator", "thank you for standing by"
        ]
        
        self.analyst_indicators = [
            "analyst", "research", "securities", "capital markets", "investment", 
            "partners", "& co", "llc", "jefferies", "morgan stanley", "jpmorgan",
            "ubs", "barclays", "goldman sachs", "deutsche bank", "credit suisse",
            "citigroup", "bank of america", "wells fargo", "evercore"
        ]
        
        self.executive_indicators = [
            "cfo", "chief financial", "chief operating", "coo", "officer", "president",
            "executive vp", "vice president", "treasurer", "finance", "operations"
        ]
        
        # Transcript artifacts to remove
        self.artifacts = [
            r"\[phonetic\]", r"\[indiscernible\]", r"\[inaudible\]", r"\[crosstalk\]",
            r"\[foreign language\]", r"\[laughter\]", r"\[applause\]", r"\[technical difficulty\]"
        ]
        
        # Speaker pattern matches various speaker formats:
        # "John Smith:" or "John Smith -" or "[John Smith]" or "JOHN SMITH:"
        self.speaker_pattern = re.compile(r"^(?:\[)?([A-Za-z\.\s-]+(?:\s[A-Za-z\.\s-]+)*)(?:\])?(?:\s*[-:]\s*|\s+)")
        
        # Patterns for identifying Q&A sections
        self.qa_pattern = re.compile(r"^\s*(?:[qQ]uestion|[qQ]\s*&?\s*[aA]|[qQ]uestions?\s+and\s+[aA]nswers?)[^A-Za-z]")
        
    def identify_ceo_name(self, file_path: str, content: str) -> str:
        """
        Attempt to identify the CEO name from the file name or content.
        
        Args:
            file_path: Path to the transcript file
            content: Content of the transcript file
            
        Returns:
            Identified CEO name or None if not found
        """
        # Try to extract from filename first (e.g., "John Doe - COMPANY.txt")
        file_name = os.path.basename(file_path)
        name_match = re.match(r"(.+?)(?:\s+-\s+|\.)", file_name)
        if name_match:
            return name_match.group(1).strip()
        
        # Look for CEO indicators in the first few lines
        first_1000_chars = content[:1000].lower()
        
        # Common CEO indicators
        ceo_indicators = [
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*[-,]\s*|\s+)(?:Chief Executive Officer|CEO|President)",
            r"((?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+)\s*[-:]\s*(?:CEO|Chief Executive Officer)",
            r"(?:CEO|Chief Executive Officer|President)\s*[-:]\s*((?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+)"
        ]
        
        for indicator in ceo_indicators:
            match = re.search(indicator, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        # If all else fails, return the first part of the filename
        return file_name.split(' - ')[0].strip()
    
    def has_speaker_indicators(self, content: str) -> bool:
        """
        Check if the transcript has explicit speaker indicators.
        
        Args:
            content: The transcript content
            
        Returns:
            True if speaker indicators are found, False otherwise
        """
        # Look for common speaker patterns (at least 3 occurrences)
        speaker_matches = re.findall(r"([A-Za-z\.\s-]+)(?:\s*[-:]\s*)", content)
        if len(speaker_matches) >= 3:
            return True
            
        # Look for Q&A format
        qa_matches = re.search(self.qa_pattern, content)
        if qa_matches:
            return True
            
        return False
    
    def is_ceo_speaking(self, speaker: str, ceo_name: str) -> bool:
        """
        Determine if the current speaker is the CEO.
        
        Args:
            speaker: The identified speaker
            ceo_name: The name of the CEO
            
        Returns:
            True if the speaker is the CEO, False otherwise
        """
        if not speaker or not ceo_name:
            return False
            
        speaker_lower = speaker.lower()
        ceo_name_lower = ceo_name.lower()
        
        # Direct match
        if speaker_lower == ceo_name_lower:
            return True
        
        # First name or last name match
        ceo_parts = ceo_name_lower.split()
        if len(ceo_parts) > 0:
            # Check if first name or last name matches
            if ceo_parts[0] == speaker_lower or ceo_parts[-1] == speaker_lower:
                return True
                
            # Check if the speaker contains the CEO's full name
            if ceo_name_lower in speaker_lower:
                return True
            
        # Check for CEO indicators
        if "ceo" in speaker_lower or "chief executive" in speaker_lower:
            return True
            
        return False
    
    def is_other_speaker(self, speaker: str) -> bool:
        """
        Determine if the speaker is someone other than the CEO.
        
        Args:
            speaker: The identified speaker
            
        Returns:
            True if the speaker is identified as not the CEO, False otherwise
        """
        if not speaker:
            return False
            
        speaker_lower = speaker.lower()
        
        # Check moderator indicators
        for indicator in self.moderator_indicators:
            if indicator in speaker_lower:
                return True
                
        # Check analyst indicators
        for indicator in self.analyst_indicators:
            if indicator in speaker_lower:
                return True
                
        # Check executive indicators (only if clearly not CEO)
        for indicator in self.executive_indicators:
            if indicator in speaker_lower and "ceo" not in speaker_lower and "chief executive" not in speaker_lower:
                return True
                
        return False
    
    def preprocess_transcript(self, content: str, ceo_name: str) -> str:
        """
        Preprocess the transcript to extract only the CEO's speech.
        
        Args:
            content: Content of the transcript file
            ceo_name: Name of the CEO
            
        Returns:
            Preprocessed transcript with only the CEO's speech
        """
        # Clean up any special characters and normalize whitespace
        content = re.sub(r'\r\n', '\n', content)
        
        # Remove transcript artifacts
        for artifact in self.artifacts:
            content = re.sub(artifact, "", content)
        
        # Check if the transcript has speaker indicators
        has_speakers = self.has_speaker_indicators(content)
        
        # If no speaker indicators found, assume the whole transcript is from the CEO
        # (after removing header/footer if present)
        if not has_speakers:
            # Try to remove headers/footers
            lines = content.split('\n')
            clean_lines = []
            content_started = False
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines at the beginning
                if not content_started and not line:
                    continue
                    
                # Skip header/title lines
                if not content_started and (ceo_name in line or re.search(r"(?:Q\d|Quarter|Earnings|Conference|Call|Transcript)", line, re.IGNORECASE)):
                    continue
                    
                # Skip footer lines
                if re.search(r"(?:Copyright|Conference Call|Call concluded|End of|Thank you)", line, re.IGNORECASE):
                    continue
                
                content_started = True
                if line:
                    clean_lines.append(line)
            
            # Add CEO name at the top
            result = [ceo_name] + clean_lines
            return '\n'.join(result)
        
        # If speaker indicators found, extract the CEO's speech
        lines = content.split('\n')
        processed_lines = []
        current_speaker = None
        is_ceo_section = False
        in_qa_section = False
        
        # Add CEO name as the first line for reference
        processed_lines.append(ceo_name)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for Q&A section markers
            if self.qa_pattern.match(line):
                in_qa_section = True
                i += 1
                continue
            
            # Check if this line indicates a new speaker
            speaker_match = self.speaker_pattern.match(line)
            if speaker_match:
                current_speaker = speaker_match.group(1).strip()
                # Remove speaker indicator from the line
                line = self.speaker_pattern.sub("", line).strip()
                
                # Determine if this is the CEO speaking
                is_ceo_section = self.is_ceo_speaking(current_speaker, ceo_name)
                
                # In Q&A sections, we need to be more careful about attribution
                if in_qa_section and not is_ceo_section and not self.is_other_speaker(current_speaker):
                    # If we can't clearly identify the speaker in Q&A, give benefit of doubt
                    # that it might be the CEO responding
                    is_ceo_section = True
                
                # Skip lines from non-CEO speakers
                if not is_ceo_section:
                    i += 1
                    continue
            
            # Only include lines when the CEO is speaking (or we think they are)
            if (is_ceo_section or current_speaker is None) and line:
                # Clean up excessive whitespace
                line = re.sub(r'\s+', ' ', line).strip()
                
                # Only add non-empty lines
                if line:
                    processed_lines.append(line)
            
            i += 1
        
        # If we didn't extract much content, fall back to the original approach
        # (this helps with transcripts that don't follow standard formats)
        if len('\n'.join(processed_lines)) < 200:  # Arbitrary threshold
            return self.preprocess_transcript_fallback(content, ceo_name)
        
        # Join the processed lines and return
        return '\n'.join(processed_lines)
    
    def preprocess_transcript_fallback(self, content: str, ceo_name: str) -> str:
        """
        Fallback preprocessing method that is less aggressive in filtering.
        Used when standard preprocessing doesn't extract enough content.
        
        Args:
            content: Content of the transcript file
            ceo_name: Name of the CEO
            
        Returns:
            Preprocessed transcript with likely CEO speech
        """
        lines = content.split('\n')
        processed_lines = [ceo_name]  # Start with CEO name
        
        # Remove obvious non-CEO parts
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious headers and footers
            if re.search(r"(?:Copyright|Conference Call|Call concluded|Operator|Moderator|Disclaimer)", line, re.IGNORECASE):
                continue
                
            # Skip lines that are clearly from analysts/questions
            if re.search(r"^\s*(?:Q:|Question:|[A-Za-z]+ from [A-Za-z]+:)", line):
                continue
                
            # Clean the line
            for artifact in self.artifacts:
                line = re.sub(artifact, "", line)
                
            line = re.sub(r'\s+', ' ', line).strip()
            
            if line:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def process_directory(self, input_dir: str, output_dir: str, ceo_name_mapping: Dict[str, str] = None):
        """
        Process all transcript files in the input directory.
        
        Args:
            input_dir: Directory containing the raw transcript files
            output_dir: Directory to save the preprocessed CEO-only transcripts
            ceo_name_mapping: Optional mapping of file names to CEO names
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all txt files in the input directory
        transcript_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        processed_count = 0
        
        print(f"Found {len(transcript_files)} transcript files. Processing...")
        
        for file_name in transcript_files:
            try:
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, f"preprocessed_{file_name}")
                
                # Read the transcript file
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Identify the CEO name
                if ceo_name_mapping and file_name in ceo_name_mapping:
                    ceo_name = ceo_name_mapping[file_name]
                else:
                    ceo_name = self.identify_ceo_name(input_path, content)
                
                # Preprocess the transcript
                processed_content = self.preprocess_transcript(content, ceo_name)
                
                # Write the preprocessed transcript
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(transcript_files)} files...")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
        
        print(f"Processing complete. {processed_count} files processed.")


def load_ceo_name_mapping(csv_path: str) -> Dict[str, str]:
    """
    Load CEO name mapping from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary mapping file names to CEO names
    """
    mapping = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    file_name = row[0].strip()
                    ceo_name = row[1].strip()
                    mapping[file_name] = ceo_name
    except Exception as e:
        print(f"Error loading CEO name mapping: {str(e)}")
    
    return mapping


def main():
    """Main function to run the preprocessing script."""
    parser = argparse.ArgumentParser(description='Preprocess CEO earnings call transcripts')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing raw transcripts')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for preprocessed transcripts')
    parser.add_argument('--ceo_name_file', type=str, help='Optional CSV file mapping transcript files to CEO names')
    
    args = parser.parse_args()
    
    # Load CEO name mapping if provided
    ceo_name_mapping = None
    if args.ceo_name_file and args.ceo_name_file != "N/A":
        ceo_name_mapping = load_ceo_name_mapping(args.ceo_name_file)
    
    # Initialize preprocessor and process directory
    preprocessor = TranscriptPreprocessor()
    preprocessor.process_directory(args.input_dir, args.output_dir, ceo_name_mapping)


if __name__ == '__main__':
    main() 