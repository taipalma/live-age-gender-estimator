#!/usr/bin/env python


import VideoThread

if __name__ == '__main__':

    help_message = '''
    USAGE: LiveDetector.py <image_names> ...
    
    Press any key to continue, ESC to stop.
    '''
     
    videoThread = VideoThread.VideoThread()
    videoThread.start()
    

