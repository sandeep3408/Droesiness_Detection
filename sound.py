# Step 2: Import necessary libraries
import streamlit as st
from playsound import playsound

# Step 3: Define Streamlit app
def main():
    st.title('Sound Alert Example')

    # Step 4: Add alert functionality
    if st.button('Sound Alert'):
        playsound('alert.mp3')  # Replace 'alert_sound.wav' with your alert sound file

if __name__ == '__main__':
    main()
