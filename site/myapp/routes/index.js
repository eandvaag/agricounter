var express = require('express');
var upload_files = require('multer')();
var router = express.Router();

let landing = require('../controllers/landing');
let socket_api = require("../socket_api");


router.get('/', landing.get_sign_in);
router.post('/', landing.post_sign_in);

router.get('/logout', landing.logout);

router.get('/admin', landing.get_admin);
router.post('/admin', landing.post_admin);

router.get('/home/:username', landing.get_home);
router.post('/home/:username', landing.post_home);

router.get('/workspace/:username/:farm_name/:field_name/:mission_date', landing.get_workspace);
router.post('/workspace/:username/:farm_name/:field_name/:mission_date', landing.post_workspace);
router.post('/workspace/:username/:farm_name/:field_name/:mission_date/annotations_upload', upload_files.array('source_file[]'), landing.post_annotations_upload);

router.post('/image_set_upload', upload_files.array('source_file[]'), landing.post_image_set_upload);
router.post('/orthomosaic_upload', upload_files.array('source_file[]'), landing.post_orthomosaic_upload);

router.get('/viewer/:username/:farm_name/:field_name/:mission_date/:result_uuid', landing.get_viewer);
router.post('/viewer/:username/:farm_name/:field_name/:mission_date/:result_uuid', landing.post_viewer);

router.post('/overlay_appearance_change/:username', landing.post_overlay_appearance_change);

router.post('/status_notification', socket_api.post_status_notification);
router.post('/upload_notification', socket_api.post_upload_notification);
router.post('/results_notification', socket_api.post_results_notification);
router.post('/model_notification', socket_api.post_model_notification);



module.exports = router;
