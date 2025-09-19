import urllib.request
import time
import os
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            self.total = total_size
            self.update(downloaded - self.n)
            # Store last block info for size calculation
            self.last_block_num = block_num
            self.block_size = block_size

def download_with_progress(url, filename):
    start_time = time.time()
    
    try:
        print(f"Starting download: {filename}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                               desc=filename, unit_divisor=1024) as t:
            urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
        
        # Calculate download speed and file size
        elapsed_time = time.time() - start_time
        speed = (t.last_block_num * t.block_size / elapsed_time) / 1024  # KB/s
        file_size_mb = (t.last_block_num * t.block_size) / (1024 * 1024)  # MB
        
        print("\nDownload completed!")
        print(f"Average speed: {speed:.2f} KB/s")
        print(f"File size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"\nDownload failed: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
            print("Incomplete file removed")
        raise

if __name__ == "__main__":
    # Example usage
    # download_url = 'https://github.com/weichenrs/REST_code/releases/download/models/REST_water_swin_large.pth'
    # output_file = 'checkpoints/REST_water_swin_large.pth'
    
    download_url = 'https://github.com/weichenrs/REST_code/releases/download/models-0.1/baseline_fbp_swin_large.pth'
    output_file = 'checkpoints/baseline_fbp_swin_large.pth'
    
    download_with_progress(download_url, output_file)