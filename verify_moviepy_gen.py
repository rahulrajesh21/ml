import os
from moviepy import ColorClip, concatenate_videoclips

def create_dummy_video(filename, duration=5):
    print(f"Creating dummy video: {filename}")
    # Create a simple video with color
    clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=duration)
    clip.fps = 24
    clip.write_videofile(filename, codec='libx264', audio=False, logger=None)
    print("Video created.")

def test_cutting(input_file, output_file):
    print(f"Testing cutting from {input_file} to {output_file}")
    from moviepy import VideoFileClip
    
    video = VideoFileClip(input_file)
    # Cut from 1s to 3s
    clip = video.subclipped(1, 3)
    clip.write_videofile(output_file, codec='libx264', audio=False, logger=None)
    video.close()
    clip.close()
    print("Cut successful.")

if __name__ == "__main__":
    try:
        dummy_file = "test_source.mp4"
        output_file = "test_output.mp4"
        
        create_dummy_video(dummy_file)
        test_cutting(dummy_file, output_file)
        
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        if os.path.exists(output_file):
            os.remove(output_file)
            
        print("✅ MoviePy verification successful!")
    except Exception as e:
        print(f"❌ MoviePy verification failed: {e}")
