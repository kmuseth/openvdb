// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <stdio.h>// for std::remove
#include <string>
#include <fstream>

#include "Tool.h"

#include "gtest/gtest.h"

// The fixture for testing class.
class Test_vdb_tool : public ::testing::Test
{
protected:
    Test_vdb_tool() {}

    ~Test_vdb_tool() override {}

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // produce vector of tokenized c-strings from a single input string
    static std::vector<char*> getArgs(const std::string &line)
    {
      const auto tmp = openvdb::vdb_tool::tokenize(line, " ");
      std::vector<char*> args;
      std::transform(tmp.begin(), tmp.end(), std::back_inserter(args),
      [](const std::string &s){
        char *c = new char[s.size()+1];
        std::strcpy(c, s.c_str());
        return c;
      });
      return args;
    }

}; // Test_vdb_tool

TEST_F(Test_vdb_tool, Util)
{
    {// findMatch
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("bc", {"abc,a", "ab,c,bc"}));
      EXPECT_EQ(4, openvdb::vdb_tool::findMatch("abc", {"abd", "cba", "ab", "abc"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findMatch("abc", {"abc", "abc ", "ab", "bc"}));
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("abc", {" abc", "abc", "ab", "abc"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findMatch("o", {"abc,o", "abc", "ab", "abc"}));
      EXPECT_EQ(3, openvdb::vdb_tool::findMatch("j", {"abc,o", "a,b,c", "ab,k,j", "abc,d,a,w"}));
      EXPECT_EQ(4, openvdb::vdb_tool::findMatch("aa", {"abc,o", "a,b,c", "ab,k,j", "abc,d,aa,w"}));
      EXPECT_EQ(2, openvdb::vdb_tool::findMatch("aaa", {"abc,o", "a,aaa,c,aa", "ab,k,j", "abc,d,bb,w"}));
    }
    {// find_all
      auto vec = openvdb::vdb_tool::find_all("%1234%678%0123%");
      EXPECT_EQ( 4, vec.size());
      EXPECT_EQ( 0, vec[0]);
      EXPECT_EQ( 5, vec[1]);
      EXPECT_EQ( 9, vec[2]);
      EXPECT_EQ(14, vec[3]);
    }
    {// to_lower_case
      EXPECT_EQ(" abc=", openvdb::vdb_tool::to_lower_case(" AbC="));
    }
    {// contains
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base"));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", "base", 5));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "base", 6));
      EXPECT_TRUE( openvdb::vdb_tool::contains("path/base.ext", 'b'));
      EXPECT_FALSE(openvdb::vdb_tool::contains("path/base.ext", "bbase"));
    }
    {// getFile
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("path/base.ext"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("/path/base.ext"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("C:\\path\\base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFile("/path/base"));
      EXPECT_EQ("base.ext", openvdb::vdb_tool::getFile("base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getFile("base"));
    }
    {// getBase
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("/path/base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("C:\\path\\base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("/path/base"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("base.ext"));
      EXPECT_EQ("base", openvdb::vdb_tool::getBase("base"));
    }
    {// getExt
      EXPECT_EQ("ext", openvdb::vdb_tool::getExt("path/file_100.ext"));
      EXPECT_EQ("ext", openvdb::vdb_tool::getExt("path/file.100.ext"));
      EXPECT_EQ("e", openvdb::vdb_tool::getExt("path/file_100.e"));
      EXPECT_EQ("", openvdb::vdb_tool::getExt("path/file_100."));
      EXPECT_EQ("", openvdb::vdb_tool::getExt("path/file_100"));
    }
     {// findFileExt
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}, false));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.eXt", {"ext", "abs", "ab"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"ext", "ext", "ab"}));
      EXPECT_EQ(3, openvdb::vdb_tool::findFileExt("path/file_002.EXT", {"e",   "ex",  "ext"}));
      EXPECT_EQ(1, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"ext", "ext", "ab"}));
      EXPECT_EQ(0, openvdb::vdb_tool::findFileExt("path/file_002.ext", {"abc", "efg", "ab"}));
    }
    {// replace
      EXPECT_EQ("base%",  openvdb::vdb_tool::replace("base%", 'i', "1"));
      EXPECT_EQ("base1234",  openvdb::vdb_tool::replace("base%i", 'i', "1234"));
      EXPECT_EQ("base%1", openvdb::vdb_tool::replace("base%1", 'i', "1"));
      EXPECT_EQ("base1",  openvdb::vdb_tool::replace("base%1i", 'i', "1"));
      EXPECT_EQ("path/base_0003.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_0003_03.vdb", openvdb::vdb_tool::replace("path/base_%4i_%2i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_0003_%2j.vdb", openvdb::vdb_tool::replace("path/base_%4i_%2j.vdb",  'i', "3"));
      EXPECT_EQ("path/base_20003_02.vdb", openvdb::vdb_tool::replace(openvdb::vdb_tool::replace("path/base_%j%4i_%2j.vdb",  'i', "3"),'j', "2"));
      EXPECT_EQ("path/base_1003.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb",  'i', "1003"));
      EXPECT_EQ("path/base_3.vdb", openvdb::vdb_tool::replace("path/base_%i.vdb",  'i', "3"));
      EXPECT_EQ("path/base_%4i.vdb", openvdb::vdb_tool::replace("path/base_%4i.vdb", 'j', "3"));
      EXPECT_EQ("path/base_0003.vdb", openvdb::vdb_tool::replace("path/base_0003.vdb", 'i', "3"));

      EXPECT_EQ("f=1.2", openvdb::vdb_tool::replace("f=%i", 'i', "1.2"));
      EXPECT_EQ("f=1.2", openvdb::vdb_tool::replace("f=%3i", 'i', "1.2"));
      EXPECT_EQ("f=01.2", openvdb::vdb_tool::replace("f=%4i", 'i', "1.2"));
      EXPECT_EQ("f=%", openvdb::vdb_tool::replace("f=%", 'i', "1.2"));
      EXPECT_EQ("f=1", openvdb::vdb_tool::replace("f=1", 'i', "1.2"));
      EXPECT_EQ("1", openvdb::vdb_tool::replace("1", 'i', "1.2"));
      EXPECT_EQ("%", openvdb::vdb_tool::replace("%", 'i', "1.2"));
    }

    {// test LoopParam
      struct LoopParam {
        char c; double value, end, stride;
        bool next() { value += stride; return value < end; }
        std::string str() const {return floor(value) == value ? std::to_string(int(value)) : std::to_string(float(value));}
      } p{'i', 0.0, 4.0, 2.0};

      EXPECT_EQ("value=0", openvdb::vdb_tool::replace("value=%i", p.c, p.str()));
      EXPECT_TRUE( p.next() );
      EXPECT_EQ("value=002", openvdb::vdb_tool::replace("value=%3i", p.c, p.str()));
      LoopParam q{'v', 0.1, 4.0, 0.5};
      EXPECT_EQ("value=0.100000", openvdb::vdb_tool::replace("value=%v", q.c, q.str()));
      EXPECT_TRUE( q.next() );
      EXPECT_EQ("value=0.600000", openvdb::vdb_tool::replace("value=%v", q.c, q.str()));
    }

    {// starts_with
      EXPECT_TRUE(openvdb::vdb_tool::starts_with("vfxvfxvfx",  "vfx"));
      EXPECT_FALSE(openvdb::vdb_tool::starts_with("vvfxvfxvfx", "vfx"));
    }

    {// ends_with
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("vfxvfxvfx",  "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("vvfxvfxvfx", "vfx"));
      EXPECT_TRUE(openvdb::vdb_tool::ends_with("file.ext", "ext"));
    }

    {// tokenize
      auto tokens = openvdb::vdb_tool::tokenize("1 2 3-4 5   6");
      EXPECT_EQ(5, tokens.size());
      EXPECT_EQ("1",   tokens[0]);
      EXPECT_EQ("2",   tokens[1]);
      EXPECT_EQ("3-4", tokens[2]);
      EXPECT_EQ("5",   tokens[3]);
      EXPECT_EQ("6",   tokens[4]);
      tokens = openvdb::vdb_tool::tokenize("1 2 3-4 5   6", " -");
      EXPECT_EQ(6, tokens.size());
      EXPECT_EQ("1",   tokens[0]);
      EXPECT_EQ("2",   tokens[1]);
      EXPECT_EQ("3",   tokens[2]);
      EXPECT_EQ("4",   tokens[3]);
      EXPECT_EQ("5",   tokens[4]);
      EXPECT_EQ("6",   tokens[5]);
    }
    {// tokenize vectors
      auto tokens = openvdb::vdb_tool::tokenize("(1,2,3)", ",()");
      EXPECT_EQ(3,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      tokens = openvdb::vdb_tool::tokenize("1,2,3", ",()");
      EXPECT_EQ(3,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      tokens = openvdb::vdb_tool::tokenize("((1,2,3),(4,5,6))", ",()");
      EXPECT_EQ(6,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      EXPECT_EQ("4", tokens[3]);
      EXPECT_EQ("5", tokens[4]);
      EXPECT_EQ("6", tokens[5]);
      tokens = openvdb::vdb_tool::tokenize("[(1,2,3),(4,5,6)]", ",()[]");
      EXPECT_EQ(6,   tokens.size());
      EXPECT_EQ("1", tokens[0]);
      EXPECT_EQ("2", tokens[1]);
      EXPECT_EQ("3", tokens[2]);
      EXPECT_EQ("4", tokens[3]);
      EXPECT_EQ("5", tokens[4]);
      EXPECT_EQ("6", tokens[5]);
    }

    {// vectorize
      auto vec = openvdb::vdb_tool::vectorize<float>("[(1.1,2.3,3.4),(4.3,5.6,6.7)]", ",()[]");
      EXPECT_EQ(   6, vec.size());
      EXPECT_EQ(1.1f, vec[0]);
      EXPECT_EQ(2.3f, vec[1]);
      EXPECT_EQ(3.4f, vec[2]);
      EXPECT_EQ(4.3f, vec[3]);
      EXPECT_EQ(5.6f, vec[4]);
      EXPECT_EQ(6.7f, vec[5]);
    }

    {// trim
      EXPECT_EQ("-a-=bs-=", openvdb::vdb_tool::trim(" -a-=bs-= "));
      EXPECT_EQ("a-=bs", openvdb::vdb_tool::trim(" -a-=bs-= ", " =-"));
    }

    {// findArg
      EXPECT_NO_THROW({
        EXPECT_EQ("bar", openvdb::vdb_tool::findArg({"v=foo", "val=bar"}, "val"));
        EXPECT_EQ("", openvdb::vdb_tool::findArg({"v=foo", "val="}, "val"));
      });
      EXPECT_THROW(openvdb::vdb_tool::findArg({"v=foo", "va=bar"}, "val"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::findArg({"v=foo", "val"}, "val"), std::invalid_argument);
    }

    {// is_int
      int i=-1;
      EXPECT_FALSE(openvdb::vdb_tool::is_int("", i));
      EXPECT_EQ(-1, i);
      EXPECT_TRUE(openvdb::vdb_tool::is_int("-5", i));
      EXPECT_EQ(-5, i);
      EXPECT_FALSE(openvdb::vdb_tool::is_int("-6.0", i));
    }

    {// str2int
      EXPECT_NO_THROW({
        EXPECT_EQ( 1, openvdb::vdb_tool::str2int("1"));
        EXPECT_EQ(-5, openvdb::vdb_tool::str2int("-5"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2int("1.0"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2int("1 "),  std::invalid_argument);
    }

    {// is_flt
      float v=-1.0f;
      EXPECT_FALSE(openvdb::vdb_tool::is_flt("", v));
      EXPECT_EQ(-1.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::is_flt("-5", v));
      EXPECT_EQ(-5.0f, v);
      EXPECT_TRUE(openvdb::vdb_tool::is_flt("-6.0", v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::is_flt("-7.0f", v));
    }

    {// str2float
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02f, openvdb::vdb_tool::str2float("0.02"));
        EXPECT_EQ( 1.0f, openvdb::vdb_tool::str2float("1"));
        EXPECT_EQ(-5.0f, openvdb::vdb_tool::str2float("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2float(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2float("1.0f"), std::invalid_argument);
    }

    {// str2double
      EXPECT_NO_THROW({
        EXPECT_EQ(0.02, openvdb::vdb_tool::str2double("0.02"));
        EXPECT_EQ( 1.0, openvdb::vdb_tool::str2double("1"));
        EXPECT_EQ(-5.0, openvdb::vdb_tool::str2double("-5.0"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2double(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2double("1.0f"), std::invalid_argument);
    }

    {// str2bool
      EXPECT_NO_THROW({
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("1"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("true"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("TRUE"));
        EXPECT_TRUE(openvdb::vdb_tool::str2bool("TrUe"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("0"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("false"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("FALSE"));
        EXPECT_FALSE(openvdb::vdb_tool::str2bool("FaLsE"));
      });
      EXPECT_THROW(openvdb::vdb_tool::str2bool(""), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("2"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("t"), std::invalid_argument);
      EXPECT_THROW(openvdb::vdb_tool::str2bool("f"), std::invalid_argument);
    }

    {// is_number
      int i=0;
      float v=0;
      EXPECT_FALSE(openvdb::vdb_tool::is_number("", i, v));
      EXPECT_EQ(0, i);
      EXPECT_EQ(1, openvdb::vdb_tool::is_number("-5",i,  v));
      EXPECT_EQ(-5, i);
      EXPECT_EQ(2, openvdb::vdb_tool::is_number("-6.0", i, v));
      EXPECT_EQ(-6.0, v);
      EXPECT_FALSE(openvdb::vdb_tool::is_number("-7.0f", i, v));
    }
}

TEST_F(Test_vdb_tool, getArgs)
{
  const std::vector<char*> args = getArgs("cmd -action option=1.0");
  EXPECT_EQ(3, args.size());
  EXPECT_EQ(0, strcmp("cmd", args[0]));
  EXPECT_EQ(0, strcmp("-action", args[1]));
  EXPECT_EQ(0, strcmp("option=1.0", args[2]));
}

TEST_F(Test_vdb_tool, Geometry)
{
  openvdb::vdb_tool::Geometry geo;
  {// test empty
    EXPECT_TRUE(geo.isEmpty());
    EXPECT_FALSE(geo.isPoints());
    EXPECT_FALSE(geo.isMesh());
  }
  {// test non-empty
    geo.setName("test");
    geo.vtx().emplace_back(1.0f, 2.0f, 3.0f);
    geo.vtx().emplace_back(4.0f, 5.0f, 6.0f);
    geo.vtx().emplace_back(7.0f, 8.0f, 9.0f);
    geo.vtx().emplace_back(10.0f, 11.0f, 12.0f);
    geo.tri().emplace_back(0,1,2);
    geo.tri().emplace_back(1,2,3);
    geo.quad().emplace_back(0,1,2,3);
    EXPECT_FALSE(geo.isEmpty());
    EXPECT_FALSE(geo.isPoints());
    EXPECT_TRUE(geo.isMesh());
    EXPECT_EQ(4, geo.vtxCount());
    EXPECT_EQ(2, geo.triCount());
    EXPECT_EQ(1, geo.quadCount());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
  {// Geometry::Header
    openvdb::vdb_tool::Geometry::Header header(geo);
    EXPECT_EQ(4, header.name);
    EXPECT_EQ(4, header.vtx);
    EXPECT_EQ(2, header.tri);
    EXPECT_EQ(1, header.quad);
  }
  std::string buffer;
  {// test streaming to buffer
    std::ostringstream os(std::ios_base::binary);
    const size_t size = geo.write(os);
    EXPECT_TRUE(size>0);
    buffer = os.str();
    EXPECT_EQ(size, buffer.size());
  }
  {// test streaming from buffer
    std::istringstream is(buffer, std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_EQ(buffer.size(), geo2.read(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
  {// write to file
    std::ofstream os("data/test.geo", std::ios_base::binary);
    EXPECT_TRUE(geo.write(os));
  }
  {// read from file
    std::ifstream is("data/test.geo", std::ios_base::binary);
    openvdb::vdb_tool::Geometry geo2;
    EXPECT_TRUE(geo2.read(is));
    EXPECT_EQ(4, geo2.vtxCount());
    EXPECT_EQ(2, geo2.triCount());
    EXPECT_EQ(1, geo2.quadCount());
    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.bbox().min());
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.bbox().max());

    EXPECT_EQ(openvdb::Vec3f(1,2,3), geo.vtx()[0]);
    EXPECT_EQ(openvdb::Vec3f(4,5,6), geo.vtx()[1]);
    EXPECT_EQ(openvdb::Vec3f(7,8,9), geo.vtx()[2]);
    EXPECT_EQ(openvdb::Vec3f(10,11,12), geo.vtx()[3]);

    EXPECT_EQ(openvdb::Vec3I(0,1,2), geo.tri()[0]);
    EXPECT_EQ(openvdb::Vec3I(1,2,3), geo.tri()[1]);

    EXPECT_EQ(openvdb::Vec4I(0,1,2,3), geo.quad()[0]);
  }
}// Geometry

TEST_F(Test_vdb_tool, Stack)
{
    using namespace openvdb::vdb_tool;
    Stack s;
    EXPECT_EQ(0, s.size());
    EXPECT_TRUE(s.empty());
    s.push("foo");
    EXPECT_EQ(1, s.size());
    EXPECT_FALSE(s.empty());
    EXPECT_EQ("foo", s.pop());
    s.push("foo");
    s.push("bar");
    EXPECT_EQ(2, s.size());
    EXPECT_FALSE(s.empty());
    s.drop();
    EXPECT_EQ(1, s.size());
    EXPECT_EQ("foo", s.top());
    EXPECT_EQ("foo", s.peek());
    s.top() = "bar";
    EXPECT_EQ("bar", s.top());
    EXPECT_EQ("bar", s.peek());
    s.dup();
    EXPECT_EQ(2, s.size());
    EXPECT_EQ(Stack({"bar", "bar"}), s);
    s.top() = "foo";
    EXPECT_EQ(Stack({"bar", "foo"}), s);
    s.swap();
    EXPECT_EQ(Stack({"foo", "bar"}), s);
    s.nip();
    EXPECT_EQ(Stack({"bar"}), s);
    s.push("foo");
    s.push("bla");
    EXPECT_EQ(Stack({"bar", "foo", "bla"}), s);
    s.scrape();
    EXPECT_EQ(Stack({"bla"}), s);
    s.push("foo");
    s.push("bar");
    EXPECT_EQ(Stack({"bla", "foo", "bar"}), s);
    s.over();
    EXPECT_EQ(Stack({"bla", "foo", "bar", "foo"}), s);
    s.top()="bob";
    EXPECT_EQ(Stack({"bla", "foo", "bar", "bob"}), s);
    s.rot();
    EXPECT_EQ(Stack({"bla", "bar", "bob", "foo"}), s);
    s.tuck();
    EXPECT_EQ(Stack({"bla", "foo", "bar", "bob"}), s);
    //s.print();
    std::stringstream ss;
    s.print(ss);
    EXPECT_EQ(std::string(" bla foo bar bob"), ss.str());
}

TEST_F(Test_vdb_tool, Translator)
{
    using namespace openvdb::vdb_tool;
    Storage s;
    Translator t(s);

    // test set and get, i.e. @ and $
    EXPECT_THROW({t("{$file}");}, std::invalid_argument);
    EXPECT_THROW({t("{dup}");},  std::invalid_argument);
    EXPECT_THROW({t("{drop}");}, std::invalid_argument);
    EXPECT_THROW({t("{swap}");}, std::invalid_argument);

    //EXPECT_NO_THROW({// everything below should pass and not throw!

    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), t("{$pi}"));
    EXPECT_TRUE(t("{path/base_0123.ext:@file}").empty());
    EXPECT_EQ("path/base_0123.ext", t("{$file}"));
    EXPECT_TRUE(t("{1:@G}").empty());
    EXPECT_EQ("1", t("{$G}"));
    EXPECT_TRUE(t("{$file:upper:@file2}").empty());
    EXPECT_EQ("PATH/BASE_0123.EXT", t("{$file2}"));
    EXPECT_TRUE(t("{$G:1000:+:@F}").empty());
    EXPECT_EQ("1001", t("{$F}"));
    EXPECT_TRUE(t("{0.1:@x:0.2:@y}").empty());
    EXPECT_EQ("0.1", t("{$x}"));
    EXPECT_EQ("0.2", t("{$y}"));
    EXPECT_TRUE(t("{1:$G:+:@G}").empty());
    EXPECT_EQ("2", t("{$G}"));
    EXPECT_TRUE(t("{$G:++:@G}").empty());
    EXPECT_EQ("3", t("{$G}"));
    EXPECT_EQ(7, s.size());

    // test file-name methods
    EXPECT_EQ("path", t("{$file:path}"));
    EXPECT_EQ("base_0123.ext", t("{$file:file}"));
    EXPECT_EQ("base_0123", t("{$file:name}"));
    EXPECT_EQ("base_", t("{$file:base}"));
    EXPECT_EQ("0123", t("{$file:number}"));
    EXPECT_EQ("ext", t("{$file:ext}"));

    EXPECT_EQ("6", t("{5:1:+}"));
    EXPECT_EQ(std::to_string(6.0f), t("{5.0:1:+}"));
    EXPECT_EQ(std::to_string(6.2f), t("{5.0:1.2:+}"));

    EXPECT_EQ("4", t("{5:1:-}"));
    EXPECT_EQ(std::to_string(4.0f), t("{5.0:1:-}"));
    EXPECT_EQ(std::to_string(3.8f), t("{5.0:1.2:-}"));

    EXPECT_EQ("10", t("{5:2:*}"));
    EXPECT_EQ(std::to_string(10.0f), t("{5.0:2:*}"));
    EXPECT_EQ(std::to_string(6.0f), t("{5.0:1.2:*}"));

    EXPECT_EQ("5", t("{10:2:/}"));
    EXPECT_EQ("0", t("{2:10:/}"));
    EXPECT_EQ(std::to_string(5.0f), t("{10.0:2.0:/}"));
    EXPECT_EQ(std::to_string(0.2f), t("{2.0:10.0:/}"));

    EXPECT_EQ("6", t("{5:++}"));
    EXPECT_EQ(std::to_string(6.2f), t("{5.2:++}"));

    EXPECT_EQ("4", t("{5:--}"));
    EXPECT_EQ(std::to_string(4.2f), t("{5.2:--}"));

    EXPECT_EQ("0", t("{5:2:==}"));
    EXPECT_EQ("0", t("{5.0:2.0:==}"));
    EXPECT_EQ("1", t("{5:5:==}"));
    EXPECT_EQ("1", t("{5.0:5.0:==}"));
    EXPECT_EQ("0", t("{foo:bar:==}"));
    EXPECT_EQ("1", t("{foo:foo:==}"));

    EXPECT_EQ("1", t("{5:2:!=}"));
    EXPECT_EQ("1", t("{5.0:2.0:!=}"));
    EXPECT_EQ("0", t("{5:5:!=}"));
    EXPECT_EQ("0", t("{5.0:5.0:!=}"));
    EXPECT_EQ("1", t("{foo:bar:!=}"));
    EXPECT_EQ("0", t("{foo:foo:!=}"));

    EXPECT_EQ("0", t("{5:2:<=}"));
    EXPECT_EQ("0", t("{5.0:2.0:<=}"));
    EXPECT_EQ("0", t("{foo:bar:<=}"));
    EXPECT_EQ("1", t("{2:5:<=}"));
    EXPECT_EQ("1", t("{2.0:5.0:<=}"));
    EXPECT_EQ("1", t("{bar:foo:<=}"));
    EXPECT_EQ("1", t("{5:5:<=}"));
    EXPECT_EQ("1", t("{5.0:5.0:<=}"));
    EXPECT_EQ("1", t("{foo:foo:<=}"));

    EXPECT_EQ("1", t("{5:2:>=}"));
    EXPECT_EQ("1", t("{5.0:2.0:>=}"));
    EXPECT_EQ("1", t("{foo:bar:>=}"));
    EXPECT_EQ("0", t("{2:5:>=}"));
    EXPECT_EQ("0", t("{2.0:5.0:>=}"));
    EXPECT_EQ("0", t("{bar:foo:>=}"));
    EXPECT_EQ("1", t("{5:5:>=}"));
    EXPECT_EQ("1", t("{5.0:5.0:>=}"));
    EXPECT_EQ("1", t("{foo:foo:>=}"));

    EXPECT_EQ("1", t("{5:2:>}"));
    EXPECT_EQ("1", t("{5.0:2.0:>}"));
    EXPECT_EQ("1", t("{foo:bar:>}"));
    EXPECT_EQ("0", t("{2:5:>}"));
    EXPECT_EQ("0", t("{2.0:5.0:>}"));
    EXPECT_EQ("0", t("{bar:foo:>}"));
    EXPECT_EQ("0", t("{5:5:>}"));
    EXPECT_EQ("0", t("{5.0:5.0:>}"));
    EXPECT_EQ("0", t("{foo:foo:>}"));

    EXPECT_EQ("0", t("{5:2:<}"));
    EXPECT_EQ("0", t("{5.0:2.0:<}"));
    EXPECT_EQ("0", t("{foo:bar:<}"));
    EXPECT_EQ("1", t("{2:5:<}"));
    EXPECT_EQ("1", t("{2.0:5.0:<}"));
    EXPECT_EQ("1", t("{bar:foo:<}"));
    EXPECT_EQ("0", t("{5:5:<}"));
    EXPECT_EQ("0", t("{5.0:5.0:<}"));
    EXPECT_EQ("0", t("{foo:foo:<}"));

    EXPECT_EQ("1", t("{0:!}"));
    EXPECT_EQ("0", t("{1:!}"));
    EXPECT_EQ("1", t("{false:!}"));
    EXPECT_EQ("0", t("{true:!}"));

    EXPECT_EQ("1", t("{0:1:|}"));
    EXPECT_EQ("1", t("{1:0:|}"));
    EXPECT_EQ("1", t("{1:1:|}"));
    EXPECT_EQ("0", t("{0:0:|}"));
    EXPECT_EQ("1", t("{false:true:|}"));
    EXPECT_EQ("0", t("{false:false:|}"));

    EXPECT_EQ("0", t("{0:1:&}"));
    EXPECT_EQ("0", t("{1:0:&}"));
    EXPECT_EQ("1", t("{1:1:&}"));
    EXPECT_EQ("0", t("{0:0:&}"));
    EXPECT_EQ("0", t("{false:true:&}"));
    EXPECT_EQ("0", t("{false:false:&}"));

    EXPECT_EQ("1", t("{1:abs}"));
    EXPECT_EQ("1", t("{-1:abs}"));
    EXPECT_EQ(std::to_string(1.2f), t("{1.2:abs}"));
    EXPECT_EQ(std::to_string(1.2f), t("{-1.2:abs}"));

    EXPECT_EQ(std::to_string(1.0f), t("{1:ceil}"));
    EXPECT_EQ(std::to_string(2.0f), t("{1.2:ceil}"));
    EXPECT_EQ(std::to_string(-1.0f), t("{-1.2:ceil}"));

    EXPECT_EQ(std::to_string(1.0f), t("{1:floor}"));
    EXPECT_EQ(std::to_string(1.0f), t("{1.2:floor}"));
    EXPECT_EQ(std::to_string(-2.0f), t("{-1.2:floor}"));

    EXPECT_EQ("4", t("{2:pow2}"));
    EXPECT_EQ(std::to_string(4.0f), t("{2.0:pow2}"));

    EXPECT_EQ("8", t("{2:pow3}"));
    EXPECT_EQ(std::to_string(8.0f), t("{2.0:pow3}"));

    EXPECT_EQ("9", t("{3:2:pow}"));
    EXPECT_EQ(std::to_string(9.0f), t("{3.0:2.0:pow}"));

    EXPECT_EQ("2", t("{3:2:min}"));
    EXPECT_EQ("-2", t("{3:-2:min}"));
    EXPECT_EQ(std::to_string(2.0f), t("{3.0:2.0:min}"));
    EXPECT_EQ(std::to_string(-2.0f), t("{3.0:-2.0:min}"));

    EXPECT_EQ("3", t("{3:2:max}"));
    EXPECT_EQ("3", t("{3:-2:max}"));
    EXPECT_EQ(std::to_string(3.0f), t("{3.0:2.0:max}"));
    EXPECT_EQ(std::to_string(2.0f), t("{-3.0:2.0:max}"));

    EXPECT_EQ("-3", t("{3:neg}"));
    EXPECT_EQ("3", t("{-3:neg}"));
    EXPECT_EQ(std::to_string(-3.0f), t("{3.0:neg}"));
    EXPECT_EQ(std::to_string(3.0f), t("{-3.0:neg}"));

    EXPECT_EQ(std::to_string(sin(2.0f)), t("{2:sin}"));
    EXPECT_EQ(std::to_string(sin(2.0f)), t("{2.0:sin}"));

    EXPECT_EQ(std::to_string(cos(2.0f)), t("{2:cos}"));
    EXPECT_EQ(std::to_string(cos(2.0f)), t("{2.0:cos}"));

    EXPECT_EQ(std::to_string(tan(2.0f)), t("{2:tan}"));
    EXPECT_EQ(std::to_string(tan(2.0f)), t("{2.0:tan}"));

    EXPECT_EQ(std::to_string(asin(2.0f)), t("{2:asin}"));
    EXPECT_EQ(std::to_string(asin(2.0f)), t("{2.0:asin}"));

    EXPECT_EQ(std::to_string(acos(2.0f)), t("{2:acos}"));
    EXPECT_EQ(std::to_string(acos(2.0f)), t("{2.0:acos}"));

    EXPECT_EQ(std::to_string(atan(2.0f)), t("{2:atan}"));
    EXPECT_EQ(std::to_string(atan(2.0f)), t("{2.0:atan}"));

    EXPECT_NEAR(openvdb::math::pi<float>(), str2float(t("{180.0:d2r}")), 1e-4);
    EXPECT_NEAR(180.0f, str2float(t("{$pi:r2d}")), 1e-4);

    EXPECT_EQ(std::to_string(1.0f/2.0f), t("{2:inv}"));
    EXPECT_EQ(std::to_string(1.0f), t("{1.0:inv}"));
    EXPECT_EQ(std::to_string(1.0f/1.2f), t("{1.2:inv}"));

    EXPECT_EQ(std::to_string(exp(1.2f)), t("{1.2:exp}"));
    EXPECT_EQ(std::to_string(log(1.2f)), t("{1.2:ln}"));
    EXPECT_EQ(std::to_string(log10(1.2f)), t("{1.2:log}"));
    EXPECT_EQ(std::to_string(sqrt(1.2f)), t("{1.2:sqrt}"));
    EXPECT_EQ("1", t("{1:int}"));
    EXPECT_EQ("1", t("{1.2:int}"));
    EXPECT_EQ(std::to_string(1.0f), t("{1:float}"));
    EXPECT_EQ(std::to_string(1.2f), t("{1.2:float}"));

    EXPECT_EQ("abcde012", t("{AbCdE012:lower}"));
    EXPECT_EQ("ABCDE012", t("{AbCdE012:upper}"));

    EXPECT_EQ("1", t("{1:dup:==}"));
    EXPECT_EQ("2", t("{1:2:nip}"));
    EXPECT_EQ("1", t("{1:2:drop}"));
    EXPECT_EQ(std::to_string(0.5f), t("{1.0:2.0:/}"));
    EXPECT_EQ(std::to_string(2.0f), t("{1.0:2.0:swap:/}"));
    EXPECT_EQ(std::to_string(2.0f/1.0f+1.0f), t("{1.0:2.0:over:/:+}"));

    EXPECT_EQ(std::to_string(2.0f/3.0f+1.0f), t("{1.0:2.0:3.0:/:+}"));
    EXPECT_EQ(std::to_string(3.0f/1.0f+2.0f), t("{1.0:2.0:3.0:rot:/:+}"));// rot(1 2 3) = 2 3 1
    EXPECT_EQ(std::to_string(1.0f/2.0f+3.0f), t("{1.0:2.0:3.0:tuck:/:+}"));// tuck(1 2 3) = 3 1 2

    EXPECT_EQ("123", t("{123:0:pad0}"));
    EXPECT_EQ("123", t("{123:1:pad0}"));
    EXPECT_EQ("123", t("{123:2:pad0}"));
    EXPECT_EQ("123", t("{123:3:pad0}"));
    EXPECT_EQ("0123", t("{123:4:pad0}"));
    EXPECT_EQ("00123", t("{123:5:pad0}"));
    EXPECT_EQ("000123", t("{123:6:pad0}"));

    EXPECT_EQ("0", t("{size}"));
    EXPECT_EQ("1", t("{0:size:scrape}"));
    EXPECT_EQ("2", t("{0:1:size:scrape}"));
    EXPECT_EQ("3", t("{0:1:2:size:scrape}"));
    EXPECT_EQ("4", t("{0:1:2:3:size:scrape}"));
    EXPECT_EQ("4", t("{0:1:2:3:clear:4}"));
    EXPECT_EQ("4", t("{0:1:2:3:size:@size:clear:$size}"));

    EXPECT_EQ("1", t("{pi:exists}"));
    EXPECT_EQ("0", t("{foo:exists}"));
    EXPECT_EQ("1", t("{8:@bar:bar:exists}"));

    EXPECT_EQ(std::to_string(sqrt(0.1f*0.1f + 0.2f*0.2f)), t("{$x:pow2:$y:pow2:+:sqrt}"));

    EXPECT_EQ("4",t("{1:2:<:if(1:3:+)}"));
    EXPECT_EQ("",t("{1:2:>:if(1:3:+)}"));
    EXPECT_EQ("1",t("{5:@a:1:2:<:if(1:@a):$a}"));
    EXPECT_EQ("5",t("{5:@a:1:2:>:if(1:@a):$a}"));

    EXPECT_EQ("4",t("{1:2:<:if(1:3:+?2:2:-)}"));
    EXPECT_EQ("0",t("{1:2:>:if(1:3:+?2:2:-)}"));
    EXPECT_EQ("1",t("{1:2:<:if(1:@a?2:@a):$a}"));
    EXPECT_EQ("2",t("{1:2:>:if(1:@a?2:@a):$a}"));
    EXPECT_EQ(std::to_string(sqrt(4+16)),t("{$pi:2:>:if(2:pow2:4:pow2:+:sqrt?2:sin)}"));
    EXPECT_EQ(std::to_string(sin(2)),t("{$pi:2:<:if(2:pow2:4:pow2:+:sqrt?2:sin)}"));

    EXPECT_EQ("a", t("{1:switch(1:a?2:b?3:c)}"));
    EXPECT_EQ("b", t("{2:switch(1:a?2:b?3:c)}"));
    EXPECT_EQ("c", t("{3:switch(1:a?2:b?3:c)}"));
    //EXPECT_THROW({t("{0:switch(1:a?2:b?3:c)}");}, std::invalid_argument);
    //EXPECT_THROW({t("{4:switch(1:a?2:b?3:c)}");}, std::invalid_argument);
    EXPECT_EQ("SUPER", t("{1:switch(1:super:upper?2:1:2:+?3:$pi)}"));
    EXPECT_EQ("3", t("{2:switch(1:super:upper?2:1:2:+?3:$pi)}"));
    EXPECT_EQ(std::to_string(openvdb::math::pi<float>()), t("{3:switch(1:super:upper?2:1:2:+?3:$pi)}"));

    // find two real roots of a quadratic polynomial
    EXPECT_EQ(" 0.683375 7.316625", t("{1:@a:-8:@b:5:@c:$b:pow2:4:$a:*:$c:*:-:@c:-2:$a:*:@a:$c:0:==:if($b:$a:/):$c:0:>:if($c:sqrt:dup:$b:+:$a:/:$b:rot:-:$a:/):squash}"));
    
    EXPECT_EQ("foo bar bla", t("{foo_bar_bla:_ :replace}"));
    EXPECT_EQ("foo_bar_bla", t("{foo bar bla: _:replace}"));
    //});// end EXPECT_NO_THROW
}// Translator

TEST_F(Test_vdb_tool, ToolParser)
{
    using namespace openvdb::vdb_tool;
    int alpha = 0, alpha_sum = 0;
    float beta = 0.0f, beta_sum = 0.0f;
    std::string path, base, ext;

    Parser p({{"alpha", "64"}, {"beta", "4.56"}});
    p.addAction("process_a", "a", "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha = p.get<int>("alpha");
                     beta  = p.get<float>("beta");}
               );
    p.addAction("process_b", "b", "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){alpha_sum += p.get<int>("alpha");
                     beta_sum  += p.get<float>("beta");}
               );
    p.addAction("process_c", "c", "docs",
              {{"alpha", "", "", ""},{"beta", "", "", ""},{"gamma", "", "", ""}},
               [&](){p.setDefaults();},
               [&](){path += (path.empty()?"":",") + p.getStr("alpha");
                     base += (base.empty()?"":",") + p.getStr("beta");
                     ext  += (ext.empty() ?"":",") + p.getStr("gamma");}
               );
    p.finalize();

    auto args = getArgs("vdb_tool -quiet -process_a alpha=128 -for v=0.1,0.4,0.1 -b alpha={$#v:++} beta={$v} -end");
    p.parse(args.size(), args.data());
    EXPECT_EQ(0, alpha);
    EXPECT_EQ(0.0f, beta);
    EXPECT_EQ(0, alpha_sum);
    EXPECT_EQ(0.0f, beta_sum);
    p.run();
    EXPECT_EQ(128, alpha);// defined explicitly
    EXPECT_EQ(4.56f, beta);// default value
    EXPECT_EQ(1 + 2 + 3, alpha_sum);// derived from loop
    EXPECT_EQ(0.1f + 0.2f + 0.3f, beta_sum);// derived from loop

    args = getArgs("vdb_tool -quiet -each file=path1/base1.ext1,path2/base2.ext2 -c alpha={$file:path} beta={$file:name} gamma={$file:ext} -end");
    p.parse(args.size(), args.data());
    p.run();
    EXPECT_EQ(path, "path1,path2");
    EXPECT_EQ(base, "base1,base2");
    EXPECT_EQ(ext,  "ext1,ext2");
}// ToolParser

TEST_F(Test_vdb_tool, ToolBasic)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -sphere r=1.1 -ls2mesh -write data/sphere.ply data/config.txt");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}// ToolBasic

TEST_F(Test_vdb_tool, Counter)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere_1.ply");
    std::remove("data/config_2.txt");

    EXPECT_FALSE(file_exists("data/sphere_1.ply"));
    EXPECT_FALSE(file_exists("data/config_2.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -eval {1:@G} -sphere r=1.1 -ls2mesh -write data/sphere_{$G}.ply data/config_{$G:++}.txt");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere_1.ply"));
    EXPECT_TRUE(file_exists("data/config_2.txt"));
}// Counter

TEST_F(Test_vdb_tool, ToolForLoop)
{
    using namespace openvdb::vdb_tool;

    std::remove("data/config.txt");
    EXPECT_FALSE(file_exists("data/config.txt"));
    for (int i=0; i<4; ++i) {
      const std::string name("data/sphere_"+std::to_string(i)+".ply");
      std::remove(name.c_str());
      EXPECT_FALSE(file_exists(name));
    }

    // test single for-loop
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for i=0,3,1 -sphere r=1.{$i} dim=128 name=sphere_{$i} -ls2mesh -write data/sphere_{$#i:++}.ply -end");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    for (int i=1; i<4; ++i) EXPECT_TRUE(file_exists("data/sphere_"+std::to_string(i)+".ply"));

    // test two nested for-loops
    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -for v=0.1,0.3,0.1 -each s=sphere_1,sphere_3 -read ./data/{$s}.ply -mesh2ls voxel={$v} -end -end -write data/test.vdb");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/test.vdb"));
}// ToolForLoop

TEST_F(Test_vdb_tool, ToolError)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    EXPECT_THROW({
      auto args = getArgs("vdb_tool -sphere bla=3 -ls2mesh -write data/sphere.ply data/config.txt -quiet");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    }, std::invalid_argument);

    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolKeep)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");
    std::remove("data/config.txt");

    EXPECT_FALSE(file_exists("data/sphere.vdb"));
    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_FALSE(file_exists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -default keep=1 -sphere r=2 -ls2mesh vdb=0 -write vdb=0 geo=0 data/sphere.vdb data/sphere.ply data/config.txt");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.vdb"));
    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}

TEST_F(Test_vdb_tool, ToolConfig)
{
    using namespace openvdb::vdb_tool;

    EXPECT_TRUE(file_exists("data"));
    std::remove("data/sphere.vdb");
    std::remove("data/sphere.ply");

    EXPECT_FALSE(file_exists("data/sphere.vdb"));
    EXPECT_FALSE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));

    EXPECT_NO_THROW({
      auto args = getArgs("vdb_tool -quiet -config data/config.txt");
      Tool vdb_tool(args.size(), args.data());
      vdb_tool.run();
    });

    EXPECT_TRUE(file_exists("data/sphere.vdb"));
    EXPECT_TRUE(file_exists("data/sphere.ply"));
    EXPECT_TRUE(file_exists("data/config.txt"));
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
