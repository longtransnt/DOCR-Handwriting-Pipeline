import axios from "axios";

function getImageList() {
    return axios.get('http://annotationnode-env.eba-iv5i9cmp.us-west-2.elasticbeanstalk.com/api/uploads')
            .then(image => {
                console.log('Image List: ', image.data);
                return image.data
            })
}
export default getImageList