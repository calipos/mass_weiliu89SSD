// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_segset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_string(fileListPath, "",
    "requires: in the txt,each line contains a jpg(img) and a bmp(mask).");
DEFINE_string(outDataDir, "./",
    "requires: the lmdb or leveldb filePath.");
DEFINE_string(outDataFile, "",
    "requires: the lmdb or leveldb filePath.");
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "segmentation",
    "The type of annotation {classification, detection}.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_segset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_outDataFile.compare("")==0 || FLAGS_fileListPath.compare("")==0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_segset");
//    return 1;
  }
  FLAGS_outDataFile=FLAGS_outDataDir+"/"+FLAGS_outDataFile;
  const bool is_color = !FLAGS_gray;
  const bool do_shuffle=FLAGS_shuffle;
  const string data_backend = FLAGS_backend;
  const string anno_type = FLAGS_anno_type;
  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  
  //const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;

  std::ifstream infile(FLAGS_fileListPath.c_str());
  std::vector<std::pair<std::string, boost::variant<std::string, std::string> > > lines;
  std::string filename;
  //int label;
  std::string labelname;
  if (anno_type == "classification") {
    LOG(FATAL)<<"the classification task is not defined in this .cpp";
  }
  else if (anno_type == "segmentation")
  {
  	while (infile >> filename >> labelname) {
      		lines.push_back(std::make_pair(filename, labelname));
    	}
  }
  if (do_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(FLAGS_outDataFile.c_str(), db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  //std::string root_folder(argv[1]);
  SegmentDatum seg_datum;
  Datum* img_datum = seg_datum.mutable_imgdata();
  //Datum* mask_datum = seg_datum.mutable_maskdata();
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status = true;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    filename = lines[line_id].first;
    if (anno_type == "classification") {
      LOG(FATAL)<<"the classification task is not defined in this .cpp";
    } 
    else if (anno_type == "segmentation") {
      labelname = boost::get<std::string>(lines[line_id].second);
      status = ReadTwoImageToSegDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, &seg_datum);
    }
    if (status == false) {
      LOG(WARNING) << "Failed to read " << lines[line_id].first;
      continue;
    }
    if (check_size) {
      if (!data_size_initialized) {
        data_size = img_datum->channels() * img_datum->height() * img_datum->width();
        data_size_initialized = true;
      } else {
        const std::string& data = img_datum->data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(seg_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
