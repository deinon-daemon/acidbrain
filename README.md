# acidbrain
accent identification inference training and eval kit

1. Follow along in the Colab notebook for a step by step of my process going through the dataset and applying a model inference pipeline to evaluate model performance on the dataset, specifically vis-à-vis the subset of speakers from the Philippines as requested
2. The notebook is separated into functional checkpoints and bulk inference/classification on 500+ audio samples of 10s & 16kHz can occur in less than a minute
3. Feel free to check out accuracy tables, classification reports, confusion matrices heatmaps, bar graphs, etc. etc. (ML data visualization goodness)
4. the accent_classification app service is a dockerization of the core processes of data load => data prep => inference => evaluation, it also caches the model in the application backend to minimize cold starts and will eventually include /train pipeline also
5. still tinkering on a few training / optimization / fine tuning experiments :)

cheers,
b
