from .data.raw.raw_data_ingest import main

video_links = ['https://www.youtube.com/watch?v=NWONeJKn6kc',]

for video_link in video_links:
    main(video_link)
