#--------------------------------------------------------------------------------------------------
#     Environment Set-up
#--------------------------------------------------------------------------------------------------

import streamlit as st
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

FIG_SIZE = (15,6)


#--------------------------------------------------------------------------------------------------
#     Layout
#--------------------------------------------------------------------------------------------------

st.set_page_config(
  page_title="Rainforest App",
  page_icon="",
  layout="centered",
  initial_sidebar_state="expanded",
  )

# No warning
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style='text-align: center; color: DarkBlue;'>Rainforest Species Detection</h1>", unsafe_allow_html=True)
st.text("")

analysis = st.sidebar.selectbox("Menu",['Predict','About'])

#--------------------------------------------------------------------------------------------------
#     Load model
#--------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
#     Predict
#--------------------------------------------------------------------------------------------------

if analysis == 'Predict':
  st.markdown("<h3 style='text-align: center;'>Insert text</h3>", unsafe_allow_html=True)
  st.text("")
  st.text("")
  st.markdown("<h3 style='text-align: left; color: CornflowerBlue;'>STEP 1 : Upload your wave</h3>",unsafe_allow_html=True)
  st.text("")

  uploaded_file = st.file_uploader("Choose a file" ,type=["wav"] )

  if uploaded_file is None:
    st.text("")
  else:
    # Load sound file
    signal, sample_rate = librosa.load(uploaded_file, sr=22050)

    #--------------------------------------
    # Audio
    #--------------------------------------

    st.audio(uploaded_file, format='audio/ogg')

    #--------------------------------------
    # Display wave
    #--------------------------------------

    disp_wave = st.checkbox('Display Wave')
    if disp_wave :
      st.markdown("<h4 style='color: DarkBlue;'>Display of the Wave</h4>", unsafe_allow_html=True)
      plt.figure(figsize=FIG_SIZE)
      lib = librosa.display.waveplot(signal, sample_rate, alpha=0.4)
      plt.xlabel("Time (s)")
      plt.ylabel("Amplitude")
      plt.title("Waveform")
      st.pyplot()

    #--------------------------------------
    # Display specto
    #--------------------------------------

    st.markdown("<h3 style='text-align: left; color: CornflowerBlue;'>STEP 2 : Transform into a Spectogram</h3>",unsafe_allow_html=True)

    # Display specto
    st.markdown("<h4 style='color: DarkBlue;'>Spectogram</h4>", unsafe_allow_html=True)

    # Parameters
    hop_length = 512
    n_fft = 2048
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate


    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    st.pyplot()

    #--------------------------------------
    # Predict Species
    #--------------------------------------

    st.markdown("<h3 style='text-align: left; color: CornflowerBlue;'>STEP 3 : Predict Species</h3>",unsafe_allow_html=True)
    if st.button('Predict this sound'):
      st.write('Prediction')

#--------------------------------------------------------------------------------------------------
#     About
#--------------------------------------------------------------------------------------------------


if analysis == 'About':
  st.text('go')
