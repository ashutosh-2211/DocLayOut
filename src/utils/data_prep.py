import os
import boto3
from datasets import load_dataset
import json
from pathlib import Path
import tempfile
from PIL import Image
import yaml
from dotenv import load_dotenv
import gc
from botocore.exceptions import NoCredentialsError

load_dotenv(override=True)

def convert_to_yolo(bboxes, category_ids, image_size):
    """Convert DocLayNet annotations to YOLO format"""
    yolo_annotations = []
    width, height = image_size
    
    # DocLayNet class mapping based on category IDs (0-10)
    
    for bbox, category_id in zip(bboxes, category_ids):
        # bbox format: [x, y, width, height] (absolute coordinates)
        # Convert to YOLO format (normalized center coordinates)
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        norm_width = bbox[2] / width
        norm_height = bbox[3] / height
        
        yolo_annotations.append({
            'class': category_id,  # Category IDs are already 0-10
            'x': x_center,
            'y': y_center,
            'w': norm_width,
            'h': norm_height
        })
    
    return yolo_annotations

def upload_file_to_s3(local_file_path, s3_key, s3_client, bucket_name):
    """Upload a single file to S3 with error handling"""
    try:
        s3_client.upload_file(str(local_file_path), bucket_name, s3_key)
        print(f"✓ Uploaded: {s3_key}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload {s3_key}: {str(e)}")
        return False

def create_dataset_config(bucket_name, s3_client):
    """Create dataset configuration for YOLO"""
    config = {
        'path': '/data',  # Container path
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': 11,  # number of classes
        'names': [
            'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
            'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
        ]
    }
    
    # Use temporary file to avoid disk clutter
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name
    
    # Upload config to S3
    success = upload_file_to_s3(temp_config_path, 'configs/dataset.yaml', s3_client, bucket_name)
    
    # Clean up temp file
    os.unlink(temp_config_path)
    
    return success

def process_split_streaming(dataset_stream, split_name, s3_client, bucket_name, 
                          max_samples=None, batch_size=10):
    """Process a dataset split in streaming mode with batching"""
    
    print(f"Processing {split_name} split in streaming mode...")
    
    # Use temporary directory for batch processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_base = Path(temp_dir)
        images_dir = temp_base / "images"
        labels_dir = temp_base / "labels"
        
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        batch_count = 0
        sample_count = 0
        
        try:
            for i, sample in enumerate(dataset_stream):
                if max_samples and i >= max_samples:
                    print(f"Reached maximum samples limit: {max_samples}")
                    break
                
                try:
                    # Generate filenames
                    image_filename = f"{i:06d}.jpg"
                    label_filename = f"{i:06d}.txt"
                    
                    image_path = images_dir / image_filename
                    label_path = labels_dir / label_filename
                    
                    # Process and save image
                    image = sample['image']
                    image.save(image_path, quality=85, optimize=True)  # Optimize for smaller file size
                    
                    # Get annotations from the new format
                    bboxes = sample.get('bboxes', [])
                    category_ids = sample.get('category_id', [])
                    
                    # Convert annotations to YOLO format
                    yolo_annotations = convert_to_yolo(bboxes, category_ids, image.size)
                    
                    # Save YOLO format labels
                    with open(label_path, 'w') as f:
                        for ann in yolo_annotations:
                            f.write(f"{ann['class']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")
                    
                    sample_count += 1
                    
                    # Upload in batches to manage memory
                    if (i + 1) % batch_size == 0:
                        print(f"Uploading batch {batch_count + 1} (samples {i+1-batch_size+1} to {i+1})")
                        
                        # Upload all files in current batch
                        for file_path in images_dir.glob("*.jpg"):
                            s3_key = f"data/processed/{split_name}/images/{file_path.name}"
                            upload_file_to_s3(file_path, s3_key, s3_client, bucket_name)
                            file_path.unlink()  # Delete after upload
                        
                        for file_path in labels_dir.glob("*.txt"):
                            s3_key = f"data/processed/{split_name}/labels/{file_path.name}"
                            upload_file_to_s3(file_path, s3_key, s3_client, bucket_name)
                            file_path.unlink()  # Delete after upload
                        
                        batch_count += 1
                        
                        # Force garbage collection
                        gc.collect()
                        
                        print(f"Batch {batch_count} completed. Memory cleared.")
                
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    # Print more details about the error for debugging
                    print(f"  Sample keys: {list(sample.keys()) if 'sample' in locals() else 'N/A'}")
                    if 'sample' in locals():
                        print(f"  Image type: {type(sample.get('image', 'N/A'))}")
                        print(f"  BBoxes length: {len(sample.get('bboxes', []))}")
                        print(f"  Category IDs length: {len(sample.get('category_id', []))}")
                    continue
            
            # Upload remaining files in the last batch
            remaining_images = list(images_dir.glob("*.jpg"))
            remaining_labels = list(labels_dir.glob("*.txt"))
            
            if remaining_images or remaining_labels:
                print(f"Uploading final batch with {len(remaining_images)} samples...")
                
                for file_path in remaining_images:
                    s3_key = f"data/processed/{split_name}/images/{file_path.name}"
                    upload_file_to_s3(file_path, s3_key, s3_client, bucket_name)
                
                for file_path in remaining_labels:
                    s3_key = f"data/processed/{split_name}/labels/{file_path.name}"
                    upload_file_to_s3(file_path, s3_key, s3_client, bucket_name)
        
        except Exception as e:
            print(f"Error in streaming processing: {str(e)}")
            return False
    
    print(f"✓ Completed {split_name} split: {sample_count} samples processed")
    return True

def download_and_prepare_data():
    """Download DocLayNet and convert to YOLO format using streaming"""
    
    # Validate environment variables
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {missing_vars}")
        return False
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3',
                                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
        
        bucket_name = os.getenv('BUCKET_NAME')
        
        # Test S3 connection
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ Connected to S3 bucket: {bucket_name}")
        except Exception as e:
            print(f"✗ Cannot access S3 bucket {bucket_name}: {str(e)}")
            return False
        
        print("Loading DocLayNet dataset in streaming mode...")
        
        # Load dataset in streaming mode to avoid memory issues
        dataset = load_dataset("ds4sd/DocLayNet-v1.2", streaming=True)
        
        # Configuration
        max_samples_per_split = int(os.getenv('MAX_SAMPLES_PER_SPLIT', '1000'))  # Configurable limit
        batch_size = int(os.getenv('BATCH_SIZE', '10'))  # Configurable batch size
        
        print(f"Configuration:")
        print(f"- Max samples per split: {max_samples_per_split}")
        print(f"- Batch size: {batch_size}")
        print(f"- Target bucket: {bucket_name}")
        
        # Process each split
        success_count = 0
        for split_name in ['train', 'val', 'test']:
            if split_name in dataset:
                print(f"\n{'='*50}")
                print(f"Starting {split_name.upper()} split processing")
                print(f"{'='*50}")
                
                success = process_split_streaming(
                    dataset[split_name], 
                    split_name, 
                    s3_client, 
                    bucket_name,
                    max_samples=max_samples_per_split,
                    batch_size=batch_size
                )
                
                if success:
                    success_count += 1
                
                # Clear memory between splits
                gc.collect()
            else:
                print(f"Warning: {split_name} split not found in dataset")
        
        # Create and upload dataset config
        print(f"\n{'='*50}")
        print("Creating dataset configuration...")
        config_success = create_dataset_config(bucket_name, s3_client)
        
        if config_success:
            success_count += 1
        
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE")
        print(f"✓ Successfully processed {success_count} components")
        print(f"✓ All data uploaded to s3://{bucket_name}/")
        print(f"{'='*50}")
        
        return True
        
    except NoCredentialsError:
        print("Error: AWS credentials not found")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_and_prepare_data()
    if not success:
        print("Script completed with errors")
        exit(1)
    else:
        print("Script completed successfully!")