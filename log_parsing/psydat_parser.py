import os
import sys
import pickle
import pandas as pd
import numpy as np


class GenericPsychoPyClass:
    """Base class for mock PsychoPy objects"""
    def __init__(self, *args, **kwargs):
        self.__dict__['_args'] = args
        self.__dict__['_kwargs'] = kwargs
        
    def __setattr__(self, key, value):
        self.__dict__[key] = value
        
    def __getattr__(self, name):
        # Return None for non-existent attributes
        return None
        
    def __setitem__(self, key, value):
        # Support dictionary-like behavior
        if not hasattr(self, '_items'):
            self.__dict__['_items'] = {}
        self.__dict__['_items'][key] = value
        
    def __getitem__(self, key):
        # Support dictionary-like access
        if hasattr(self, '_items') and key in self.__dict__['_items']:
            return self.__dict__['_items'][key]
        return None


# Create specialized mock classes
class ExperimentHandler(GenericPsychoPyClass):
    """Mock for psychopy.data.ExperimentHandler"""
    pass

class TrialHandler(GenericPsychoPyClass):
    """Mock for psychopy.data.TrialHandler"""
    pass


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle PsychoPy objects"""
    def __init__(self, file_obj):
        super().__init__(file_obj)
        
        # Register common PsychoPy classes
        self.known_classes = {
            'psychopy.data.experimenthandler.ExperimentHandler': ExperimentHandler,
            'psychopy.data.trial.TrialHandler': TrialHandler,
            'psychopy.data.TrialHandler': TrialHandler
        }
        
    def find_class(self, module, name):
        """Override find_class to handle missing PsychoPy modules"""
        # Check if it's a known PsychoPy class
        full_name = f"{module}.{name}"
        if full_name in self.known_classes:
            return self.known_classes[full_name]
            
        # Handle specific PsychoPy modules
        if module.startswith('psychopy'):
            # Add this class to known classes for future reference
            self.known_classes[full_name] = GenericPsychoPyClass
            return GenericPsychoPyClass
            
        # For non-PsychoPy classes, try the standard approach
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # If we can't find the class, create a generic one
            print(f"Creating generic class for {module}.{name}")
            self.known_classes[full_name] = GenericPsychoPyClass
            return GenericPsychoPyClass


def extract_data_using_dill():
    """Try to extract using dill if pickle fails"""
    try:
        import dill
        print("Using dill as a backup unpickler...")
        
        # Define function to load with dill
        def dill_load(file_path):
            with open(file_path, 'rb') as f:
                return dill.load(f)
                
        return dill_load
    except ImportError:
        print("Dill not installed. To install: pip install dill")
        return None


def safe_load_psydat(file_path):
    """Try multiple methods to load a .psydat file"""
    
    # Attempt method 1: Standard pickle
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded with standard pickle")
        return data
    except Exception as e:
        print(f"Standard pickle loading failed: {e}")
    
    # Attempt method 2: Custom unpickler
    try:
        with open(file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            data = unpickler.load()
        print("Successfully loaded with custom unpickler")
        return data
    except Exception as e:
        print(f"Custom unpickler failed: {e}")
    
    # Attempt method 3: Try with dill if available
    dill_loader = extract_data_using_dill()
    if dill_loader:
        try:
            data = dill_loader(file_path)
            print("Successfully loaded with dill")
            return data
        except Exception as e:
            print(f"Dill loading failed: {e}")
    
    # If all methods fail, try a binary approach
    print("All standard methods failed. Attempting binary analysis...")
    try:
        # Read raw file data
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Check for common patterns in PsychoPy data files
        if b'psychopy' in file_data:
            print("File contains 'psychopy' references. Most likely a proper .psydat file that requires psychopy package.")
            print("Consider installing psychopy in a separate environment: pip install psychopy")
            
        # Look for data patterns that might be useful
        if b'extraInfo' in file_data:
            print("Found 'extraInfo' in the file data - might contain experiment metadata")
            
        if b'data' in file_data and b'trials' in file_data:
            print("Found 'data' and 'trials' references - likely contains experiment trial data")
            
        # Last resort - just create a dummy object with some basic attributes
        print("Creating minimal mock object to attempt data extraction")
        mock_data = GenericPsychoPyClass()
        
        # Add some common attributes in case they're accessed
        mock_data.extraInfo = {}
        mock_data.data = {}
        mock_data.entries = []
        mock_data.trialList = []
        
        return mock_data
        
    except Exception as e:
        print(f"Binary approach failed: {e}")
        return None


def extract_attributes(obj, prefix='', max_depth=3, current_depth=0, visited=None):
    """Extract attributes from a complex object into a flat dictionary"""
    if visited is None:
        visited = set()
    
    # Avoid circular references
    obj_id = id(obj)
    if obj_id in visited or current_depth > max_depth:
        return {}
    
    visited.add(obj_id)
    results = {}
    
    # Handle different object types
    if isinstance(obj, dict):
        # For dictionaries, extract keys and values
        for key, value in obj.items():
            # Skip very large values or non-serializable types
            if isinstance(value, (str, int, float, bool, type(None))):
                results[f"{prefix}{key}"] = value
            elif isinstance(value, (list, tuple)) and len(value) < 1000:
                results[f"{prefix}{key}"] = value
    else:
        # Try to access attributes via different methods
        try:
            # Method 1: Use __dict__ if available
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if attr_name.startswith('_'):
                        continue
                    
                    # Store basic types directly
                    if isinstance(attr_value, (str, int, float, bool)) or attr_value is None:
                        results[f"{prefix}{attr_name}"] = attr_value
                    elif isinstance(attr_value, (list, tuple)) and len(attr_value) < 1000:
                        results[f"{prefix}{attr_name}"] = attr_value
                    elif isinstance(attr_value, dict):
                        # Extract from nested dictionaries
                        nested_results = extract_attributes(
                            attr_value, 
                            prefix=f"{prefix}{attr_name}.", 
                            max_depth=max_depth,
                            current_depth=current_depth + 1,
                            visited=visited
                        )
                        results.update(nested_results)
        except Exception:
            pass
        
        # Method 2: Try using dir() for attributes not in __dict__
        try:
            for attr_name in dir(obj):
                # Skip already processed attributes, private attrs, and methods
                if (attr_name in results or 
                    attr_name.startswith('_') or 
                    attr_name in ('copy', 'parent', 'origin') or
                    callable(getattr(obj, attr_name, None))):
                    continue
                
                try:
                    attr_value = getattr(obj, attr_name)
                    # Store basic types
                    if isinstance(attr_value, (str, int, float, bool)) or attr_value is None:
                        results[f"{prefix}{attr_name}"] = attr_value
                    elif isinstance(attr_value, (list, tuple)) and len(attr_value) < 1000:
                        results[f"{prefix}{attr_name}"] = attr_value
                except Exception:
                    pass
        except Exception:
            pass
    
    return results


def identify_trial_data(data_obj):
    """Find and extract trial data from a psydat object"""
    trials_data = []
    
    # Strategy 1: Check if data_obj is a list of dictionaries/objects
    if isinstance(data_obj, list) and len(data_obj) > 0:
        print("Data object is a list, checking if items are trial data...")
        
        # Check the first item to see if it looks like trial data
        first_item = data_obj[0]
        if isinstance(first_item, dict) or hasattr(first_item, '__dict__'):
            # This might be a list of trials
            for item in data_obj:
                if isinstance(item, dict):
                    trials_data.append(item.copy())  # Use copy to avoid modifying original
                else:
                    # Extract attributes
                    trial_dict = extract_attributes(item)
                    if trial_dict:
                        trials_data.append(trial_dict)
    
    # Strategy 2: Check for an 'entries' attribute (common in newer PsychoPy)
    if not trials_data and hasattr(data_obj, 'entries'):
        print("Found 'entries' attribute, examining for trial data...")
        
        entries = data_obj.entries
        if isinstance(entries, (list, tuple)) and len(entries) > 0:
            for entry in entries:
                trial_dict = extract_attributes(entry)
                if trial_dict:
                    trials_data.append(trial_dict)
    
    # Strategy 3: Check for 'trialList' and 'data' attributes (common in older PsychoPy)
    if not trials_data and hasattr(data_obj, 'trialList') and hasattr(data_obj, 'data'):
        print("Found 'trialList' and 'data' attributes...")
        
        trial_list = data_obj.trialList
        data_dict = data_obj.data
        
        if (isinstance(trial_list, (list, tuple)) and 
            isinstance(data_dict, dict) and 
            len(trial_list) > 0):
            
            for i, trial in enumerate(trial_list):
                trial_dict = {}
                
                # Get trial parameters
                if isinstance(trial, dict):
                    trial_dict.update(trial)
                else:
                    trial_dict.update(extract_attributes(trial))
                
                # Add recorded data
                for data_name, values in data_dict.items():
                    if isinstance(values, (list, tuple)) and i < len(values):
                        trial_dict[data_name] = values[i]
                
                trials_data.append(trial_dict)
    
    # Strategy 4: Check for 'thisExp' attribute (might contain the experiment handler)
    if not trials_data and hasattr(data_obj, 'thisExp'):
        print("Found 'thisExp' attribute, checking for trial data...")
        
        # Try recursively with the thisExp object
        return identify_trial_data(data_obj.thisExp)
    
    # Strategy 5: Last resort - look for lists with similar lengths
    if not trials_data:
        print("No standard trial structure found. Looking for data patterns...")
        
        # Extract all attributes
        all_attrs = extract_attributes(data_obj)
        
        # Find lists of similar length
        lists_by_length = {}
        for key, value in all_attrs.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                length = len(value)
                if length not in lists_by_length:
                    lists_by_length[length] = []
                lists_by_length[length].append(key)
        
        # Sort lengths by number of attributes (most attributes first)
        sorted_lengths = sorted(lists_by_length.keys(), 
                              key=lambda k: len(lists_by_length[k]), 
                              reverse=True)
        
        # Try the length with the most attributes first
        for length in sorted_lengths:
            attrs = lists_by_length[length]
            if length > 0 and len(attrs) > 1:
                print(f"Found {len(attrs)} attributes of length {length}")
                
                # Create trials from these attributes
                for i in range(length):
                    trial_dict = {}
                    for attr in attrs:
                        value_list = all_attrs[attr]
                        if i < len(value_list):
                            trial_dict[attr] = value_list[i]
                    trials_data.append(trial_dict)
                break
    
    return trials_data


def parse_psydat_file(file_path):
    """Parse a PsychoPy .psydat file and extract useful data"""
    try:
        # Try to load the file with our different methods
        data = safe_load_psydat(file_path)
        if data is None:
            print("Failed to load the .psydat file with any method")
            return None
        
        # Extract experiment info if available
        exp_info = {}
        if hasattr(data, 'extraInfo') and isinstance(data.extraInfo, dict):
            exp_info = data.extraInfo
            print("Experiment info found:")
            for key, value in exp_info.items():
                print(f"  {key}: {value}")
        
        # Try to find trial data
        trials_data = identify_trial_data(data)
        
        if not trials_data:
            print("No trial data identified in the file")
            return None
        
        print(f"Found {len(trials_data)} trials")
        
        # Add experiment info to each trial
        for trial in trials_data:
            for key, value in exp_info.items():
                if key not in trial:
                    trial[key] = value
        
        # Create DataFrame
        df = pd.DataFrame(trials_data)
        
        # Clean up the DataFrame
        # 1. Remove columns that are all None/NaN
        df = df.dropna(axis=1, how='all')
        
        # 2. Convert complex objects to strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x
                )
        
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"Error parsing .psydat file: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_to_csv(df, output_path):
    """Save DataFrame to CSV file with error handling"""
    if df is not None and not df.empty:
        try:
            # Handle NaN values
            df = df.fillna('')
            
            # Save to CSV
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Data saved to {output_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            
            # Try alternative approach with problematic columns removed
            try:
                print("Trying alternative approach...")
                # Identify columns with problematic data
                problem_cols = []
                for col in df.columns:
                    try:
                        # Test if this column can be converted to CSV
                        pd.DataFrame({col: df[col]}).to_csv(os.devnull)
                    except:
                        problem_cols.append(col)
                
                if problem_cols:
                    print(f"Removing {len(problem_cols)} problematic columns")
                    clean_df = df.drop(columns=problem_cols)
                    clean_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"Data saved to {output_path} (some columns removed)")
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
    else:
        print("No data to save")


def process_psydat_files(input_path, output_dir=None):
    """Process .psydat files from a file or directory"""
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process files
    if os.path.isfile(input_path):
        if input_path.endswith('.psydat'):
            print(f"Processing file: {input_path}")
            df = parse_psydat_file(input_path)
            
            if df is not None:
                # Create output path
                base_name = os.path.basename(input_path).replace('.psydat', '.csv')
                output_path = os.path.join(output_dir or os.path.dirname(input_path), base_name)
                
                # Save to CSV
                save_to_csv(df, output_path)
        else:
            print(f"Not a .psydat file: {input_path}")
            
    elif os.path.isdir(input_path):
        # Find all .psydat files
        psydat_files = [f for f in os.listdir(input_path) if f.endswith('.psydat')]
        print(f"Found {len(psydat_files)} .psydat files in {input_path}")
        
        # Process each file
        for filename in psydat_files:
            file_path = os.path.join(input_path, filename)
            print(f"\nProcessing file: {filename}")
            
            df = parse_psydat_file(file_path)
            
            if df is not None:
                # Create output path
                base_name = filename.replace('.psydat', '.csv')
                output_path = os.path.join(output_dir or input_path, base_name)
                
                # Save to CSV
                save_to_csv(df, output_path)
    else:
        print(f"Path not found: {input_path}")


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python psydat_parser.py input_path [output_directory]")
        print("  input_path: Path to a .psydat file or directory containing .psydat files")
        print("  output_directory (optional): Directory to save the output CSV files")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_psydat_files(input_path, output_dir)


if __name__ == "__main__":
    main()