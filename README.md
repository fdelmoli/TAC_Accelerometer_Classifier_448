# Predicting Heavy Drinking on College Campuses: A Diagnostic Review of Random Forest Model and Investigation into Alternative Approaches of Preemptive Notifications.

By: Sophia Tuinman, Colin Wang, Francesca Delmolino

### Data Processing  
We restructured our accelerometer data (`all_accelerometer_data_pids_13.csv`) and reconciled it with the TAC data (`clean_tac.csv`) of the 13 viable participants. After processing the data, we assigned binary classifications to each reading [0 (not-intoxicated), 1 (intoxicated)] where intoxicated is when TAC >= 0.08. Lastly, with this processed data, We used upsampling to fill in the sporadic data readings to fill in those gaps. Our method was: a participant is sober until their first intoxicated reading, and their intermediate synthetic readings will have the intoxicated classification until their next non-intoxicated reading from the TAC monitors (the non-synthetic data).   

Last Observation Carried Forward (LOCF): Replaces missing values with the last known value. Works well for data with rising or constant trends, but can distort trends if they change direction.

![alt text](http://url/to/img.png)

## Balancing Data
Originally the dataset was about ~82.5% sober, what we did was then extract a subset of the data so that for each participant we had ~60% sober and ~30% intoxicated.

### Feature Creation
As our next step, we used our data to generate features. To do this, we grouped our data into 4-second windows and labeled each window with the last TAC label. For each window, we calculated mean, standard deviation, median, zero crossing rate, and absolute max features for each of the x, y, z, and magnitude (except for zero crossing rate), while the spectral features were calculated on the fast Fourier transforms conducted on x + y + z of the accelerometer data.

These features, and their corresponding labels, were used as inputs for our baseline random forest model and gradient boosting model. For our training and testing of our data, we used 13-folds where each iteration isolated a participant’s data to be used for testing. This allows for our models to learn from a substantial amount of data, capturing patterns and trends present in the time series.

### Random Forest
Our Random Forest Baseline produced an average accuracy of 81%, surpassing the performance of the original study (77.5%). However, upon further investigation, we found that it struggled in accurately predicting intoxicated ratings. To remedy this, we worked to balance the data by sampling less sober data that may not be predictive.  To assess its performance rigorously, we implemented leave-one-out cross-validation with 13 folds, systematically designating one participant's data as the test set in each iteration. Notably, we obtained a notably high accuracy score of 90% with participant MC7070 as our test subject. We attribute this success to the more balanced ratio of intoxicated versus sober readings in MC7070's data, highlighting the importance of data balance in achieving accurate predictions. 

### Gradient Boosting
Additionally, we used RandomizedSearchCV() to tune hyperparameters, which did not end up improving the general Boosting model. Additionally, attempts to optimize hyperparameters using RandomizedSearchCV() did not yield significant improvements in the overall accuracy of the Gradient Boosting model. Notably, a specific configuration of hyperparameters (learning_rate=0.3, n_estimators=100, max_depth=5, subsample=0.8, random_state=42) led to worse overall accuracy, as the model tended to predict all instances as sober. Overall, Gradient Boosting produced a best accuracy of 88.52% on test participant SA0297, with a 99.49% accuracy for sober observations and 55.29% accuracy for intoxicated participants. This was a drastic improvement from other participants which tend to have less than 5% accuracy when predicting intoxicated observations.

### CNNs
Convolutional neural networks capture local patterns in time series data, which is why we initially thought this model would perform well. Unfortunately, our first few attempts with raw accelerometer data produced accuracies of 0.0%. This was due to the fact that the data inputted had covered large durations of time (several hours), which meant the model was trying to capture global patterns. So, we changed our approach by segmenting the time series data into smaller windows of time – several seconds. However, we were still unable to accurately predict TAC. We considered converting the accelerometer data into an image. At this point we had sunk a lot of time into the model and it made more sense to try a different model than to keep restructuring the data that was fed into the CNN model.

### LSTMs
LSTM models, being structured around Recurrent Neural Networks (RNNs), offer an inherent advantage in handling sequential data. Leveraging 49 observation sequences for training with the final sequence observation reserved for testing (50 observation sequences total), we encountered challenges related to significant RAM requirements, prompting exploration into utilizing Host Virtual Machines (VMs) for computational efficiency. Presently, our loss and validation loss metrics stand at 0.0381 (`LSTM_Python_Attempt.py`), indicating promising model performance, although further analysis is warranted to fully comprehend its implications.

### Feature Importance

### Model Analysis and Success
