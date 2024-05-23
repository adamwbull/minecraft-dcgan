// Profile.js

import React, { useState, useContext, useEffect } from 'react';
import { View, Text, TextInput, Pressable } from 'react-native';
import styles from '../styles/Profile';
import { darkColors } from '../styles/Colors';
import Header from './shared/Header';
import userContext from '../Context.js';
import { updatePassword } from '../API'; // Import the new function we will create in API.js
import { useLinkTo, Link } from '@react-navigation/native';

function Profile({ navigation, route }) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const user = useContext(userContext);

  const linkTo = useLinkTo()

  useEffect(() => {
    
    setTimeout(() => {
      if (user == null) {
        linkTo('/home')
      }
    }, 100)

  }, [])

  const handleSubmit = async () => {
    if (newPassword !== confirmNewPassword) {
      alert("New passwords don't match.");
      return;
    }
    console.log('user:',user)
    // Call the updatePassword API function
    const success = await updatePassword(user.token, currentPassword, newPassword);
    if (success) {
      alert('Password updated successfully.');
      setCurrentPassword('')
      setNewPassword('')
      setConfirmNewPassword('')
    } else {
      alert('Failed to update password.');
    }
  };

  return (
    <View style={styles.container}>
      <Header routeName={'/profile'} navigation={navigation} />
      <View style={styles.landing}>
        <Text style={styles.title}>Profile</Text>
        <Text style={styles.description}>Update Password</Text>
        <View style={styles.section}>
          <View style={{flex:1,marginTop:10}}>
            <TextInput 
                placeholder='Current Password...' 
                style={styles.profileTextInput}
                secureTextEntry={true}
                value={currentPassword}
                onChangeText={setCurrentPassword}
                placeholderTextColor="#ccc"
            />
            <TextInput 
                placeholder='New Password...' 
                style={styles.profileTextInput}
                secureTextEntry={true}
                value={newPassword}
                onChangeText={setNewPassword}
                placeholderTextColor="#ccc"
            />
            <TextInput 
                placeholder='Confirm New Password...' 
                style={styles.profileTextInput}
                secureTextEntry={true}
                value={confirmNewPassword}
                onChangeText={setConfirmNewPassword}
                placeholderTextColor="#ccc"
            />
            <Pressable 
              style={(newPassword.length == 0 || newPassword != confirmNewPassword) ? [styles.profileSubmitButton,{backgroundColor:darkColors.lightGray}] : styles.profileSubmitButton} 
              onPress={handleSubmit}
              disabled={newPassword.length <= 7 || newPassword != confirmNewPassword}
            >
              <Text style={(newPassword.length <= 7 || newPassword != confirmNewPassword) ? [styles.profileSubmitButtonText,{color:darkColors.black}] : styles.profileSubmitButtonText}>Update Password</Text>
            </Pressable>
            {((newPassword.length > 0 && newPassword.length <= 7) || newPassword != confirmNewPassword) && (<View style={[styles.errorBox,{padding:10}]}>
              <Text style={styles.errorBoxText}>New password must match and be at least 8 characters long.</Text>
            </View>)}
          </View>
        </View>
      </View>
    </View>
  );
}

export default Profile;
