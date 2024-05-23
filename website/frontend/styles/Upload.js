import { StyleSheet, Dimensions } from 'react-native';
import GlobalStyles from './GlobalStyles';
import { darkColors } from './Colors'

const screenWidth = Dimensions.get('window').width;

const LocalStyles = {
    container: {
        backgroundColor:darkColors.thirdBackground,
        flex:screenWidth < 768 ? null : 1,
        width:'100%'
    },
    middleSection: {
        alignItems:'center',
        flex:1
    },  
    uploadSection: {
        padding:20,
        width:screenWidth < 768 ? '100%' : '50%',
        backgroundColor:darkColors.background,
    },
    uploadTitle: {
        fontSize:24,
        fontFamily:'RobotoMono',
        color:darkColors.white
    },
    uploadButtonText: {
        color:darkColors.white,
        textAlign:'center',
        fontFamily:'RobotoMono'
    },
    uploadStipulations: {
        marginTop:20,
        marginBottom:10
    },
    uploadStipulationTitle: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    uploadStipulationText: {
        fontSize:14,
        color:darkColors.text,
        fontFamily:'RobotoMono',
        marginLeft:5,
        marginTop:5
    },
    fileUploadSection: {
    },
    fileRow: {
        flexDirection:'row',
        justifyContent:'space-between',
        alignItems:'center',
    },
    fileNameWrapper: {},
    fileName: {
        fontSize:16,
        color:darkColors.text,
        fontFamily:'RobotoMono'
    },
    fileStatus: {},
    dropzoneWrapper: {
        borderWidth:2,
        borderRadius:10,
        borderStyle:'dashed',
        borderColor:darkColors.white,
        padding:40,
        justifyContent:'center',
        alignItems:'center'
    },
    uploadTitleFiles: {
        fontSize:24,
        fontFamily:'RobotoMono',
        color:darkColors.white,
        paddingBottom:10,
        marginTop:10,
        marginBottom:20,
        borderBottomWidth:1,
        borderColor:darkColors.white
    }
};

// Merge global and local styles
const styles = { ...GlobalStyles, ...LocalStyles };

export default StyleSheet.create(styles);
