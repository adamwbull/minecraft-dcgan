import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState, useCallback, useContext } from 'react';
import { Linking, Animated, Image, StyleSheet, Text, View, Pressable } from 'react-native';
import { useLinkTo, Link } from '@react-navigation/native';
import styles from '../styles/Login';
import {darkColors} from '../styles/Colors'
import { TextInput } from 'react-native-web';
import { set, ttl } from '../Storage.js'
import { loginCheck } from '../API.js'
import userContext from '../Context.js'

export default function Welcome() {
    
  const user = useContext(userContext)
  const linkTo = useLinkTo()

  const [userData, setUserData] = useState(user)

  const [refreshing, setRefreshing] = useState(false)

  // Form data.
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  // Control data.
  const [showError, setShowError] = useState(false)

  useEffect(() => {
    
    setTimeout(() => {
      if (userData != null) {
        linkTo('/home')
      }
    }, 100)

  }, [refreshing])

  const updateEmail = (t) => {
    setShowError(false)
    setEmail(t)
  }

  const updatePassword = (t) => {
    setShowError(false)
    setPassword(t)
  }

  const submitLoginTrigger = async () => {
    console.log('Logging in...')
    setRefreshing(true)
    var check = await loginCheck(email, password)
    if (check.success) {
      console.log('User:',check.user)
      set('User', check.user, ttl)
      setTimeout(() => {
        window.location.reload()
      }, 100)
    } else {
      setShowError(true)
    }
    setRefreshing(false)
  }

  return (<View style={styles.container}>
    <View style={styles.logInContainer}>
      <View style={styles.logInStripeTop}></View>
      <View style={styles.logIn}>
        <Text style={styles.logInTitle}>Research Login</Text>
        {showError && (<View style={styles.errorBox}>
          <Text style={styles.errorBoxText}>There was a problem logging you in!{'\n'}Please check your credentials and try again.</Text>
        </View>)}
        <View style={styles.inputGroup}>
          <TextInput 
            placeholder={'Email...'}
            style={styles.logInTextInput} 
            value={email}
            onChangeText={updateEmail}
            placeholderTextColor={darkColors.lightGray}
          />
          <TextInput 
            placeholder={'Password...'} 
            style={styles.logInTextInput}
            value={password} 
            onChangeText={updatePassword}
            secureTextEntry={true}
            placeholderTextColor={darkColors.lightGray}
          />
        </View>
        <Pressable 
          style={styles.logInSubmitButton} 
          onPress={submitLoginTrigger}
          disabled={email.length == 0 || password.length < 8}
        >
          <Text style={styles.logInSubmitButtonText}>Log In</Text>
        </Pressable>
      </View>
      <View style={styles.logInStripeBottom}></View>
    </View>
  </View>)
}