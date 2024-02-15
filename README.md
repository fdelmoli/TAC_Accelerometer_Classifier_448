# Predicting Heavy Drinking on College Campuses: A Diagnostic Review of Random Forest Model and Investigation into Alternative Approaches of Preemptive Notifications.

By: Sophia Tuinman, Colin Wang, Francesca Delmolino

### Data Processing  
Colin restructured our accelerometer data (`all_accelerometer_data_pids_13.csv`) and reconciled it with the TAC data (`clean_tac.csv`) of the 13 viable participants. After processing the data, He assigned binary classifications to each reading [0 (not-intoxicated), 1 (intoxicated)] where intoxicated is when TAC >= 0.08. Lastly, with this processed data, Colin used upsampling to fill in the sporadic data readings to fill in those gaps. Our method was: a participant is sober until their first intoxicated reading, and their intermediate synthetic readings will have the intoxicated classification until their next non-intoxicated reading from the TAC monitors (the non-synthetic data).   

Last Observation Carried Forward (LOCF): Replaces missing values with the last known value. Works well for data with rising or constant trends, but can distort trends if they change direction.

### Feature Creation
