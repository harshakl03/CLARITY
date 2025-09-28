def count_images_in_folders(dataset_path):
    """Count images in each subfolder"""
    
    dataset_path = Path(dataset_path)
    total_images = 0
    
    print(f"\n🖼️  Image count by folder:")
    print("-" * 30)
    
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            # Count image files (png, jpg, jpeg)
            image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            image_count = 0
            
            for ext in image_extensions:
                image_count += len(list(folder.glob(f"**/*{ext}")))
            
            print(f"{folder.name:.<20} {image_count:>8,} images")
            total_images += image_count
    
    print("-" * 30)
    print(f"{'TOTAL':.<20} {total_images:>8,} images")
    
    return total_images

# Count images
total_count = count_images_in_folders(r"D:\Projects\CLARITY\Dataset")