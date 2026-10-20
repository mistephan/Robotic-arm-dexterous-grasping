from wrc_demo.skills.library import SkillLibrary


def test_bilingual_skill_library_retrieval(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add("soft fruit", "fruit bruised", "香蕉 青柠 mango", "Use tactile stop.")
    assert "Use tactile stop" in lib.relevant("拿取香蕉并放入篮子")[0]


def test_unrelated_guidance_is_not_loaded(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add("basket", "rim collision", "藤篮 basket", "Clear the rim.")
    assert lib.relevant("wave hello") == []
