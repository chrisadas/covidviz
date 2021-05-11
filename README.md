# covidviz

> Generate single-image COVID dashboard for Thailand

The script download [combined data](https://github.com/djay/covidthailand#combine-) from https://github.com/djay/covidthailand and generate plots that I personally want to see each day. No more, no less.

![Wide image](https://covidviz.s3-ap-southeast-1.amazonaws.com/full_figure.png?)

The script will also generate narrow image that fits into phone screen without scrolling. (my phone anyway)

![Wide image](https://covidviz.s3-ap-southeast-1.amazonaws.com/mobile_figure.png?)

I use cron to run this regularly and push the image to S3.
