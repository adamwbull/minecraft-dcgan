// Header.js styles

import { StyleSheet, Dimensions } from 'react-native';
import { darkColors } from './Colors'


const screenWidth = Dimensions.get('window').width;


const GlobalStyles = {
    container: {
        padding:20,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor:darkColors.background 
    },
    description: {
        textAlign: 'center',
        marginBottom: 20,
        fontSize: 18,
    },
    button: {
        padding: 20,
        borderRadius: 20,
        alignItems: 'center'
    },
    buttonText: {
        fontSize: 18,
    },
    infoBox: {
        padding:20,
        borderRadius:20,
        width:'100%'
    },
    notifWrapper: {
        alignItems:'flex-end',
        marginBottom:20
      },
      bubble: {
        padding:20,
        borderTopLeftRadius:20,
        borderBottomLeftRadius:20,
        flexDirection:'row',
        alignItems:'center',
        flex:1
      },
      bubbleWrapper: {
        flexDirection:'row'
      },
      bubbleButton: {
        padding:20,
        borderBottomRightRadius:20,
        borderTopRightRadius:20,
        justifyContent:'center',
        alignItems:'center',
        flexDirection:'row'
      },
      bubbleButtonText: {
        fontWeight:'bold',
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
      },
      bubbleTextWrapper: {
        flex:1,
        marginRight:20
      },
      bubbleText: {
       
      },
      navLeft: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        flex:screenWidth < 768 ? null : 1,
        marginTop:screenWidth < 768 ? 20 : 0,
        marginBottom:screenWidth < 768 ? 20 : 0
      },
      navRight: {
        flex:screenWidth < 768 ? null : 1,
        alignItems:screenWidth < 768 ? null : 'flex-end',
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        justifyContent:screenWidth < 768 ? null : 'flex-end'
      },
      navRow: {
        flexDirection:screenWidth < 768 ? 'column' : 'row',
        padding:20,
        backgroundColor:darkColors.thirdBackground,
        alignItems:'space-between',
        justifyContent:'space-between',
        width:'100%'
      },
      navRowSelected: {
        padding:20,
        backgroundColor:darkColors.blue,
        marginRight:screenWidth < 768 ? 0 : 20
      },
      navRowSelectedText: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
      },
      navRowDeselected: {
        padding:20,
        backgroundColor:darkColors.gray,
        marginRight:screenWidth < 768 ? 0 : 20,
        marginBottom:screenWidth < 768 ? 10 : 0
      },
      navRowDeselectedText: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        textAlign:'center'
      },
      blockStatsWrapper: {
        flexWrap:'wrap',
        marginTop:10,
        height:450
      },
      errorBox: {
        backgroundColor:darkColors.red,
        padding:5,
        borderRadius:10,
        marginTop:10,
      },
      errorBoxText: {
        fontSize:16,
        color:darkColors.white,
        fontFamily:'RobotoMono',
        textAlign:'center'
      },
      headerButton: {
      }
};

export default StyleSheet.create(GlobalStyles);
