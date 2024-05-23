import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
      container: {
          backgroundColor:darkColors.background,
          flex:1,
          width:'100%',
          alignItems:'center'
      },
      button: {
          padding:20,
          borderRadius:10,
          backgroundColor:darkColors.darkBlue,
          marginRight:screenWidth < 768 ? 0 : 20,
          marginTop:20,
          justifyContent:'center',
          alignItems:'center',
          width:screenWidth < 768 ? screenWidth-100 : 400
      },
      buttonText: {
          fontSize:20,
          color:darkColors.text,
          fontFamily:'RobotoMono',
          textAlign:'center'
      },
      title: {
          fontSize:24,
          color:darkColors.text,
          fontFamily:'RobotoMono'
      },
      description: {
          fontSize:20,
          color:darkColors.text,
          fontFamily:'RobotoMono',
          marginTop:20
      },
      landing: {
          padding:20,
          backgroundColor:darkColors.secondBackground,
          borderRadius:10,
          width:screenWidth < 768 ? screenWidth-60 : screenWidth/1.5,
          marginTop:20
      },
      row: {
          flexDirection:screenWidth < 768 ? 'column' : 'row',
      },
      profileContainer: {
        width: 500,
        borderRadius: 10,
        backgroundColor: darkColors.background
      },
      profileStripeTop: {
        height: 10,
        backgroundColor: darkColors.thirdBackground,
        borderTopLeftRadius: 10,
        borderTopRightRadius: 10,
        marginBottom: 10
      },
      profileStripeBottom: {
        height: 10,
        backgroundColor: darkColors.thirdBackground,
        borderBottomLeftRadius: 10,
        borderBottomRightRadius: 10,
        marginTop: 10
      },
      profile: {
        padding: 20,
      },
      logo: {
        width: 200,
        height: 200,
        margin: 20,
      },
      profileTitle: {
        fontSize: 26,
        fontFamily: 'RobotoMono',
        textAlign: 'center',
        color: darkColors.white
      },
      inputGroup: {
        marginTop: 20,
        marginBottom: 10
      },
      profileTextInput: {
        backgroundColor: darkColors.thirdBackground,
        borderRadius: 10,
        padding: 10,
        marginBottom: 10,
        color:darkColors.white
      },
      profileSubmitButton: {
        borderRadius: 10,
        backgroundColor: darkColors.green,
        padding:10,
      },
      profileSubmitButtonText: {
        fontFamily: 'RobotoMono',
        textAlign: 'center',
        color: darkColors.white,
        fontSize:16
      },
      forgotPasswordText: {
        fontSize: 18,
        fontFamily: 'RobotoMono',
        color: darkColors.white,
        textAlign: 'center',
        marginTop: 20
      },
      forgotPasswordLink: {
        fontSize: 18,
        fontFamily: 'RobotoMono',
        color: darkColors.blue,
        marginLeft: 5,
      },
      section: {
        flexDirection:'row'
      },
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
