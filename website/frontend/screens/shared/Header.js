// Header.js screen

import React, { useState, useEffect, useContext } from 'react';
import { Image, Text, View, Pressable, ActivityIndicator, ScrollView, Dimensions, Animated, TextInput, Switch  } from 'react-native';
import { locations } from '../../App'
import styles from '../../styles/GlobalStyles';
import userContext from '../../Context.js';
import { set, ttl } from '../../Storage.js'
import { darkColors } from '../../styles/Colors'
import { AntDesign } from '@expo/vector-icons'

const screenWidth = Dimensions.get('window').width;

export default function Header({ routeName, navigation }) {

  const user = useContext(userContext)

  const [loggedIn, setLoggedIn] = useState(false)

  const logout = () => {

    set('User', null, ttl)
    location.reload()

  }

  useEffect(() => {

    console.log('user:',user)

    if (user != null && user != undefined) {
      setLoggedIn(true)
    }

  }, [])

  const [showNavMobile, setShowNavMobile] = useState(false)

  return (<View style={styles.navRow}>
        {screenWidth < 768 && (<Pressable style={styles.headerButton} onPress={() => setShowNavMobile(!showNavMobile)}>
          <AntDesign 
            name={showNavMobile ? 'menu-unfold' : 'menu-fold'} 
            size={30} 
            color={darkColors.white} 
          />
        </Pressable>)}
        {(showNavMobile || screenWidth >= 768) && (<View style={styles.navLeft}>
          {locations.map((location, index) => {

            let display = false
            
            // If login is required and requirements are met...
            if (location.loginRequired && user != null && user.type >= location.minLoginTypeRequired) {
              display = true
            // Or if login is not required...
            } else if (!location.loginRequired) {
              display = true
            }

            if (display) {
              if (routeName.includes(location.routeName)) {

                return (<View style={styles.navRowSelected} key={'navRow'+index}>
                  <Text style={styles.navRowSelectedText}>{location.name}</Text>
                </View>)
  
              } else {
  
                return (<Pressable style={styles.navRowDeselected} key={'navRow'+index} onPress={() => navigation.navigate(location.name)}>
                  <Text style={styles.navRowDeselectedText}>{location.name}</Text>
                </Pressable>)
  
              }
            }

          })}
        </View>)}
        {(showNavMobile || screenWidth >= 768) && (<View style={styles.navRight}>
          <View>
            {loggedIn && (<Pressable style={styles.navRowDeselected} onPress={() => navigation.navigate('Profile')}>
                  <Text style={styles.navRowDeselectedText}>Profile</Text>
                </Pressable>)}
          </View>
          <View>
            {!loggedIn && (<Pressable style={[styles.navRowDeselected,{marginRight:0}]} onPress={() => navigation.navigate('Login')}>
                  <Text style={styles.navRowDeselectedText}>Log In</Text>
                </Pressable>) || (<Pressable style={[styles.navRowDeselected,{marginRight:0}]} onPress={() => logout()}>
                  <Text style={styles.navRowDeselectedText}>Log Out</Text>
                </Pressable>)}
          </View>
        </View>)}
      </View>)

}