# acidbrain
accent identification notebook & restful inference app service
the original sought of this task was to evaluate the performance of a given model (warisqr7/accent-id-commonaccent_xlsr-en-english)
at the task of identifying the "accent" // nationality of the speaker, using the Speech Accent Archive as our test dataset.

The Speech Accent Archive covers a broader range of speakers (geographically) than the given model is trained to decode to --
i.e. the vocabulary of output labels of the model i have to test necessarily constrained the subset of data i utilized for evaluation and training experiments.

the model has been trained to classify // distinguish between these accents:
- us
- england
- australia
- indian
- canada
- bermuda
- scotland
- african
- ireland
- newzealand
- wales
- malaysia
- philippines
- singapore
- hongkong
- southatlandtic

i also visualized and explored the Speech Archive since native_language seemed to me an equally if not greater determinant of perceived accent when speaking a non-native language.

after data prep, cleaning, and resampling i ran parallelized batch inference on a T4 gpu, achieving >0.33s it/s on >500 audio samples, each sample being roughly 20-30s of the same passage read aloud by different speakers.
the findings of accuracy by class (assigned accent label) and native_language are visualized and output below the runtime, and i scoped out below that how one might continue to use speechbrain to finetune the wav2vec encoder on Speech Accent Archive, potentially improving its performance
at the task at hand by Over Sampling data from the target class for improvement.

Inside accent_classifier is the dockerized fastapi server to run inference and testing / eval on Speech Accent Archive data included in /test_data
to replicate the service simply:
cd accent_classifier/app
docker compose up --build
(open docker desktop beforehand and now navigate to the container)
*begin docker exec terminal session inside linux container*
pytest
... tests will run ...

you can also send .wav and .mp3 files to /predict and /predict-batch to test the server via curl // HTTPS call

cheers,
b
