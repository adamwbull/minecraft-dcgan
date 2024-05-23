import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: darkColors.secondBackground
      },
      logInContainer: {
        width: 500,
        borderRadius: 10,
        backgroundColor: darkColors.background
      },
      logInStripeTop: {
        height: 10,
        backgroundColor: darkColors.thirdBackground,
        borderTopLeftRadius: 10,
        borderTopRightRadius: 10,
        marginBottom: 10
      },
      logInStripeBottom: {
        height: 10,
        backgroundColor: darkColors.thirdBackground,
        borderBottomLeftRadius: 10,
        borderBottomRightRadius: 10,
        marginTop: 10
      },
      logIn: {
        padding: 20,
      },
      logo: {
        width: 200,
        height: 200,
        margin: 20,
      },
      logInTitle: {
        fontSize: 26,
        fontFamily: 'RobotoMono',
        textAlign: 'center',
        color: darkColors.white
      },
      inputGroup: {
        marginTop: 20,
        marginBottom: 10
      },
      logInTextInput: {
        backgroundColor: darkColors.secondBackground,
        borderRadius: 10,
        padding: 10,
        marginBottom: 10,
        color:darkColors.white
      },
      logInSubmitButton: {
        borderRadius: 10,
        backgroundColor: darkColors.green,
        padding:10,
      },
      logInSubmitButtonText: {
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
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
