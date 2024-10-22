Replace audio in video files :-

1. open command prompt (run as administrator)
2. change to the project's directory --> "C:\Users\Harsh\Desktop\Video_Audio_Replacer_App\Montreal-Forced-Aligner"

	cd C:\Users\Harsh\Desktop\Video_Audio_Replacer_App\Montreal-Forced-Aligner

3. activate conda environment, in which Montreal-Forced-Aligner and its dependencies are installed

	conda activate mfa_env

4. clear all the components of these directories :  "C:\Users\Harsh\Documents\MFA\my_corpus", "C:\Users\Harsh\Desktop\Video_Audio_Replacer_App\Montreal-Forced-Aligner\output" and 

5. the final video gets stored at "C:\Users\Harsh\Desktop\Video_Audio_Replacer_App\Montreal-Forced-Aligner\output"
6. dictionary path : "C:\Users\Harsh\Documents\MFA\pretrained_models\dictionary\english_us_mfa.dict"
7. pretrained model path : "C:\Users\Harsh\Documents\MFA\pretrained_models\acoustic\english_mfa"
8. run appw8.py file (works on browser, as it is a streamlit interface)

	streamlit run appw8.py

9. a browser window will open, browse through your system and select the video file whose audio you wish to replace
10. click the button "Replace Audio" on the browser
11. let the program run for about 5 to 10 minutes
12. python files with w in their names' ends, like "appw1.py", "appw2.py", "appw3.py", etc. indicate working codes, i.e., they have been tested and work fine
13. stop the code in the command prompt by pressing "Ctrl+C"
14. close the browser window of the streamlit app only after stopping the app by "Ctrl+C"
15. when finished, deactivate the conda environment

	conda deactivate

