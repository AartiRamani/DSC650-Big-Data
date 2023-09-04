---
title: Assignment 1

subtitle: Computer performance, reliability, and scalability calculation

author: Aarti Ramani

---

## 1.2 

#### a. Data Sizes

| Data Item                                  | Size per Item | 
|--------------------------------------------|--------------:|
| 128 character message.                     | 128 Bytes     |
| 1024x768 PNG image                         | 1.5 MB        |
| 1024x768 RAW image                         | 1.127 MB      | 
| HD (1080p) HEVC Video (15 minutes)         | 160.18 MB     |
| HD (1080p) Uncompressed Video (15 minutes) | 160,181 MB    |
| 4K UHD HEVC Video (15 minutes)             | 641 MB        |
| 4k UHD Uncompressed Video (15 minutes)     | 640,723 MB    |
| Human Genome (Uncompressed)                | 0.6 GB        |


- 128 character message = 128 *1 = 128, since 1 Character = 8 bits(1 byte).


- PNG Image: 
  Formula for PNG Image Size in bytes: PNG Image Size (bytes) = (Width x Height x Bits per pixel) / Compression Ratio
  
  Bits per pixel = 16 bit
    
  pixel count = width of image in pixels × height of image in pixels
  
  pixel count = 1024 * 768 = 786432. 
  
  image file size = pixel count × bit depth
  
  imaage file size = 786432 * 16 = 12,582,912 
  
  image file size = 12,582,912 × (1 byte / 8 bits) × (1/1024) = 2304 KB * (1/1024) => 1.5 MB (uncompressed). 
  
  This can change based on the compression ratio.

 
- The size of a RAW image depends on the bit depth and sensor resolution. 
  
  Let's assume: Bit Depth: 12 bits per channel (common for many RAW formats) and No compression.

  pixel count = 1024 * 768 = 786432.
  
  impage file size = 786432 * 12 = 9,437,184
  
  9,437,184 × (1 byte / 8 bits) × (1/1024) = 1152 KB x (1/1024) = 1.125 MB
  


- HD (1080p) HEVC Video (15 minutes) :
  
  Uncompressed Video Size (bytes) = (Width x Height x Bit Depth x Frame Rate) x Duration (seconds)
  
  Raw Video Bitrate = 30 fps x 1920 pixels x 1080 pixels x 24 bits = 1,492,992,000 bps (bits per second)
  
  Uncompressed Video Size  = Raw Video Bitrate  x Duration (seconds) / 8  => 1,492,992,000 x 900 seconds/8 = 167,961,600,000
  
  Apply the compression ratio of 1000:
  
  Compressed Video Size (GB) = Uncompressed Video Size (GB) / Compression Ratio
  
  Compressed Video Size =  167,961,600,000 / 1000 = 167,961,600 => 167,961,600 *(1/1024) * (1/1024) = 160.18 MB
  
  

- HD (1080p) Uncompressed Video (15 minutes):   
  
  Uncompressed Video Size (bytes) = (Width x Height x Bit Depth x Frame Rate) x Duration (seconds)
  
  Raw Video Bitrate = 30 fps x 1920 pixels x 1080 pixels x 24 bits = 1,492,992,000  bps (bits per second)
  
  Uncompressed Video Size (GB) = Raw Video Bitrate (Gbps) x Duration (seconds) / 8  
  
  => 1,492,992,000 x 900 seconds/8 = 167,961,600,000
  
  Uncompressed Video Size = 160,180.6640625 MB = ~ 160,181 MB
  
  
- 4K UHD HEVC Video (15 minutes):
  
  Video file size = Time (second) x Frames per Second (FPS) x Pixels per Frame (Resolution) x Bit Depth
  
  Video Resolution: 3840x2160 pixels (4K UHD)
  
  Bit Depth: 24 bits 
  
  Frame Rate: 30 fps
  
  Duration: 900 seconds (15 minutes) 
  
  Uncompressed Video Size (bytes) =  30 * (3840 x 2160) * 24  = 5,971,968,000
  
  Uncompressed Video Size (GB) = Raw Video Bitrate (Gbps) x Duration (seconds) / 8  
  => 5,971,968,000 * 900 / 8 = 671,846,400,000 => 671,846,400,000 
  
  Apply the compression ratio of 1000:
  
  Compressed Video Size = Uncompressed Video Size / Compression Ratio
  
  Compressed Video Size = 671,846,400,000 / 1000 =>  671,846,400 => 640.722 MB = ~641 MB
 
 
 
- 4k UHD Uncompressed Video (15 minutes):
  
  Video file size = Time (second) x Frames per Second (FPS) x Pixels per Frame (Resolution) x Bit Depth
  
  Video Resolution: 3840x2160 pixels (4K UHD)
  
  Bit Depth: 24 bits 
  
  Frame Rate: 30 fps
  
  Duration: 900 seconds (15 minutes) 
  
  Uncompressed Video Size (bytes) = 30 * (3840 x 2160) * 24 = 5,971,968,000 
  Uncompressed Video Size (GB) = Raw Video Bitrate (Gbps) x Duration (seconds) / 8  
  => 5,971,968,000 * 900 / 8 = 671,846,400,000 => 671,846,400,000 = 640,722.6 MB = ~ 640,723 MB
  
  
  

- Uncompressed Genome Size (bytes) = Number of Base Pairs x Size per Base Pair
  
  The human genome, which is typically measured in base pairs (bp), consists of approximately 3.2 billion base pairs. This 
  
  measurement includes all the DNA in a complete human genome. In terms of file size, DNA base pairs are typically represented 
  
  using two bits per base pair (A, T, G, C), which translates to 0.2 bytes per base pair (since 1 byte = 8 bits).
  
  Uncompressed Genome Size (bytes) = 3,200,000,000 base pairs x 0.2 bytes/base pair => 640,000,000 bytes => 640 MB => 0.6 GB
 

#### b. Scaling

|                                           | Size     | # HD  | 
|-------------------------------------------|---------:|-----: |
| Daily Twitter Tweets (Uncompressed)       | 64 GB    |  1    |
| Daily Twitter Tweets (Snappy Compressed)  | 42.6 GB  |  1    |
| Daily Instagram Photos                    | 113 TB   |  34   |
| Daily YouTube Videos                      | 461 TB   | 138   |
| Yearly Twitter Tweets (Uncompressed)      | 23 TB    |  7    |
| Yearly Twitter Tweets (Snappy Compressed) | 15 TB    |  5    |
| Yearly Instagram Photos                   | 41245 TB | 12374 |
| Yearly YouTube Videos                     |168192 TB | 50458 |
 

- Total Size (bytes) = Number of Tweets (X) x Average Tweet Length (Y) x Bytes per character (UTF-8 variable-length encoding)
  
  On an average, there are around 500 million tweets per day. Assuming thetweet length is 128 characters long.
  
  500,000,000 * 128 * 1 (byte) = 64,000,000,000 bytes / (1024 MB/GB) / 1024 = 64 GB

- Snappy typically achieves compression ratios in the range of 1.5x to 1.7x for plain text data, meaning the compressed size is   about 50% to 70% of the original uncompressed size. Using 1.5, 64 GB / 1.5 = 42.66 GB 

- Daily Instagram photos:  
  
  Average Photo Size = 1.5 MB    
  
  Daily Photo Size (bytes) = Daily Photo Uploads x Average Photo Size (1024 * 768)
  
  Daily Photo Size (bytes) = 75,000,000 photos x 1.5 MB/photo   #Calculating for the 75% alone.
  
  Daily Photo Size ≈ 112,500,000 MB = 112.5 TB = ~113TB
  112.5 * 0.3 = 33.75 HDD
  
  
 
- Daily YouTube Videos: 500 hours of video is uploaded to YouTube every minute. 
  
  500 hours * 60 mins = 30000 mins
  15 mins - 160.18 MB
  (30000 mins * 160.18 )/15 = 320,360MB = 320 GB  
  320 * 24 * 60 = 460800 GB in a day = 460.8 TB 
  
  About 720000 * 4 * 160.18 MB = 461,318,400 MB (439.94 GB).
  
- 64 GB * 365 = 23,360 GB = 23 TB.

- 43 GB * 365 = 15,695 GB = 15 TB.

- 113 TB * 365 = 41,245 TB.

- 460.8 TB * 365 = 168,192 TB.

    
#### c. Reliability
|                                    | # HD | # Failures |
|------------------------------------|-----:|-----------:|
| Twitter Tweets (Uncompressed)      | 7    |  0.8617    |
| Twitter Tweets (Snappy Compressed) | 5    |  0.6155    |
| Instagram Photos                   |12374 |  1523      |
| YouTube Videos                     |50458 |  6211      |

The failure rate used is 12.31% AFR for 10TB drives Q2 2023
Failures = HD * Failure_Rate

7 * 0.1231 = 0.8617
5 * 0.1231 = 0.6155
12374 * 0.1231 = 1,523.2394
50458 * 0.1231 = 6,211.3798

#### d. Latency

|                           | One Way Latency      |
|---------------------------|---------------------:|
| Los Angeles to Amsterdam  | 30 ms                |
| Low Earth Orbit Satellite | 40 ms                |
| Geostationary Satellite   | 240 - 280 ms         |
| Earth to the Moon         | 1.3 s                |
| Earth to Mars             | 21 minutes           | 


- The distance between LA and Amsterdam is 8934 km/5551 mi, and the speed of light is about 300,000 km/sec. 
  Time = 8934 * 1000/300000 = 29.78 ms = ~30 ms.
  153.186ms -> https://wondernetwork.com/pings
  
- Low Earth Orbit Satellite = 40ms 
  https://www.omniaccess.com/leo/#:~:text=MEO%20latency%20is%20180%20milliseconds,if%20you%20are%20conducting%20High

- Geostationary Staellite latency = 240 - 280 ms https://www.satsig.net/latency.htm

- Earth to moon latency today - 1.3 s https://www.spaceacademy.net.au/spacelink/commdly.htm

- Earth to Mars Latency today - 3 - 21 minutes https://www.spaceacademy.net.au/spacelink/commdly.htm