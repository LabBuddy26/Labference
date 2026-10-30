"""Regression checks for the four-family / ordinary-task presentation split."""
import hashlib
import json
import tempfile
import re
from html import unescape
from pathlib import Path
from html.parser import HTMLParser
from unittest.mock import patch
import yaml

from eai4s_eval.cli.export_public_site import export_public_site
from eai4s_eval.public_site import (
    REPOSITORY_ROOT, build_public_snapshot, build_release_leaderboard,
    load_release_protocols,
    load_public_site_config,
)
from eai4s_eval.server.app import create_app


def test_paper_abstract_matches_report_and_branding_is_consistent():
    expected = (Path(__file__).parent / 'fixtures/labference_report_v1_abstract.txt').read_text().strip()
    with tempfile.TemporaryDirectory() as directory:
        client = create_app(outputs_root=Path(directory)).test_client()
        home = client.get('/').get_data(as_text=True)
        paragraph = re.search(r'data-paper-abstract><p>(.*?)</p>', home, re.S)
        assert paragraph and unescape(paragraph.group(1)) == expected
        assert 'class="soft paper-abstract"' in home
        assert 'class="container abstract-shell"' in home
        assert 'class="content abstract-copy"' in home
        assert 'class="tag-row abstract-keywords" aria-label="Keywords"' in home
        for route in ('/', '/tasks/', '/objects/', '/dataset/', '/leaderboard/', '/docs/'):
            page = client.get(route).get_data(as_text=True)
            assert '<strong>Labference</strong>' in page
            assert 'Chemistry manipulation benchmark' not in page
            assert '>LabBench<' not in page
        assert 'Appendix A' in home
        assert 'Device operation is an additional selected workflow' in home
        assert 'diagnostic readouts' in client.get('/leaderboard/').get_data(as_text=True)


SPECIAL_OBJECTS = {"feedback-scale", "feedback-weight-set", "feedback-sample",
                   "special-centrifuge", "rack-recall-rack", "rack-recall-tube"}


AUTHOR_NAMES = ['Shitong Liu', 'Jingwei Zhang', 'Sheng Lian', 'Xinyu Zhuang',
                'Kaiwen Guo', 'Quan Liu', 'Chunhai Zhao', 'Jiaxin He', 'Fan Yang',
                'Manli Tao', 'Tingyu Yuan', 'Jinqiao Wang', 'Xiaojing Liu',
                'Chaomin Guo', 'Feichan Yang', 'Hongwei Shi', 'Zhuang Xiong', 'Chaoyang Zhao', 'Zhiyong Liu']
NEW_PARTNER_LOGOS = ('pku-stomatology-logo-v3.png', 'pla-general-hospital-logo-v2.jpg',
                     'iquotient-robotics-logo-v2.png', 'pku-pharmaceutical-sciences-logo-v2.png',
                     'cas-semiconductors-logo-v2.png')


def test_institutions_are_separate_from_project_and_authors_match_citation():
    config = load_public_site_config()
    assert [p['name'] for p in config['partners']] == [
        '中国科学院自动化研究所', '中国科学院大学', '中科视语', '紫东太初',
        '北京大学口腔医学院（口腔医院）', '中国人民解放军总医院', '智科特机器人',
        '北京大学药学院', '中国科学院半导体研究所']
    authors = config['paper']['authors']
    assert [a['given_names'] + ' ' + a['family_names'] for a in authors] == AUTHOR_NAMES
    assert [a['given_names'] + ' ' + a['family_names'] for a in authors if a['corresponding']] == [
        'Shitong Liu', 'Chaoyang Zhao', 'Zhiyong Liu']
    assert [a['affiliations'] for a in authors] == [[1, 2]] * 4 + [[1]] * 8 + [[3], [4], [5], [6], [7], [1], [1, 2]]
    assert all(a['given_names'] and a['family_names'] for a in authors)
    affiliations = {a['id']: a['name'] for a in config['paper']['affiliations']}
    semiconductor_name = ('State Key Laboratory of Semiconductor Physics and Chip Technologies, '
                          'Institute of Semiconductors, Chinese Academy of Sciences, Beijing, P.R. China')
    assert affiliations[7] == semiconductor_name
    partner_name = 'Institute of Semiconductors, Chinese Academy of Sciences'
    assert config['partners'][-1]['name_en'] == partner_name
    assert config['partners'][-1]['short_name'] == partner_name
    cff = yaml.safe_load((REPOSITORY_ROOT / 'docs/public/CITATION.cff').read_text())
    assert [a['given-names'] + ' ' + a['family-names'] for a in cff['authors']] == AUTHOR_NAMES
    assert [a['affiliation'] for a in cff['authors']] == [
        '; '.join(affiliations[i] for i in a['affiliations']) for a in authors]
    with tempfile.TemporaryDirectory() as directory:
        client = create_app(outputs_root=Path(directory)).test_client()
        page = client.get('/').get_data(as_text=True)
        assert page.index('class="project-mark"') < page.index('id="institutions"') < page.index('id="abstract"')
        institutions = page.split('id="institutions"', 1)[1].split('</section>', 1)[0]
        assert institutions.count('data-institution') == 9
        assert 'eai4s-labbench-mark.webp' not in institutions
        assert '中国科学院半导体研究所' in institutions
        assert partner_name in institutions
        assert semiconductor_name not in institutions
        assert 'cas-semiconductors-logo-v2.png' in institutions
        assert 'cas-semiconductors-logo.webp' not in institutions
        assert 'id="authors"' not in page
        assert 'href="#authors"' not in page
        assert 'data-author-name=' not in page
        assert '† Corresponding authors' not in page
        assert 'author-affiliation-' not in page
        bib = unescape(re.search(r'<pre id="bibtex">(.*?)</pre>', page, re.S).group(1))
        bib_authors = re.search(r'author=\{ (.*?) \}', bib).group(1).split(' and ')
        assert bib_authors == AUTHOR_NAMES


def test_empty_author_list_hides_section_and_navigation():
    config = load_public_site_config()
    config['paper']['authors'] = []
    with tempfile.TemporaryDirectory() as directory:
        client = create_app(outputs_root=Path(directory)).test_client()
        with patch('eai4s_eval.public_site.load_public_site_config', return_value=config):
            page = client.get('/').get_data(as_text=True)
        assert 'id="authors"' not in page
        assert 'href="#authors"' not in page
        assert 'Corresponding authors' not in page


class MediaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        for key in ("src", "poster", "data-src"):
            value = attributes.get(key, "")
            if "/static/" in value:
                self.references.append(value)


def test_home_has_only_four_special_tasks_and_two_videos():
    with tempfile.TemporaryDirectory() as directory:
        client = create_app(outputs_root=Path(directory)).test_client()
        page = client.get("/").get_data(as_text=True)
        assert '<a href="#abstract">Abstract</a>' in page
        assert '<h2 class="section-title">Abstract</h2>' in page
        assert page.index('class="hero"') < page.index('id="abstract"') < page.index('id="special-tasks"')
        assert 'L01–L12' in page and 'R01–R12' in page
        assert 'Labference' in page
        assert 'chemistry laboratory' not in page.lower()
        assert 'Matched Oracle-assisted continuations' in page
        assert page.count('class="tag-item"') == 6
        for name, logo in [('中科视语', 'objecteye-logo-v2.jpg'), ('紫东太初', 'zidong-taichu-logo-v2.jpg')]:
            assert name in page
            assert f'/static/web_assets/brand/{logo}' in page
        assert page.count('data-special-family=') == 4
        assert page.count('data-public-status="implemented"') == 2
        assert page.count('data-public-status="pending"') == 2
        assert page.count('<video ') == 2
        assert '2 implemented' not in page and '2 pending' not in page
        assert 'Pending implementation' not in page and 'COMING SOON' not in page
        for scene in ('centrifuge-scene.webp', 'rack-recall-scene.webp'):
            assert f'/static/web_assets/task_media/special/{scene}' in page
        assert 'data-demo-card' not in page
        assert 'data-featured-track' not in page
        assert 'labbench-family' not in page
        assert 'weighing-hard-front.mp4' in page
        assert 'device-operation-hard-front.mp4' in page
        parser = MediaParser()
        parser.feed(page)
        assert all(client.get(reference).status_code == 200 for reference in parser.references)


def test_task_collections_are_separate_and_specials_come_first():
    with tempfile.TemporaryDirectory() as directory:
        page = create_app(outputs_root=Path(directory)).test_client().get('/tasks/').get_data(as_text=True)
        assert page.index('id="release-protocols"') < page.index('id="ordinary-tasks"')
        assert page.count('data-special-family=') == 4
        assert page.count('data-task-card ') == 24
        assert page.count('data-track="long_horizon"') == 12
        assert page.count('data-track="reasoning_in_action"') == 12
        assert 'data-task-search' in page and 'data-task-track' in page
        assert '2 implemented' not in page and 'Pending implementation' not in page


def test_public_status_does_not_change_runtime_or_model_score_contracts():
    runs = [
        {"protocol_id": "weighing_hard", "model_name": "Test model", "checkpoint": "v1",
         "ranking_eligible": True, "evaluation_valid": True, "scores": {"valid_success": True}},
        {"protocol_id": "device_operation_hard", "model_name": "Test model", "checkpoint": "v1",
         "ranking_eligible": True, "evaluation_valid": True, "scores": {"valid_success": False}},
        {"protocol_id": "centrifuge_hard", "model_name": "Reference", "checkpoint": "debug",
         "ranking_eligible": False, "scores": {"valid_success": True}},
    ]
    original = build_release_leaderboard(runs)
    snapshot = build_public_snapshot([], release_runs=runs)
    grouped = snapshot['release_leaderboard']
    assert original['models'][0]['overall_score_100'] == grouped['models'][0]['provisional_score_100']
    assert grouped['models'][0]['overall_score_100'] is None
    assert grouped['models'][0]['rank'] is None
    assert grouped['models'][0]['family_coverage'] == {'evaluated': 2, 'total': 4}
    assert grouped['models'][0]['task_coverage']['total'] == 4
    assert snapshot['summary']['special_task_count'] == 4
    assert snapshot['summary']['task_count'] == 24
    assert all(not board['models'] for board in snapshot['leaderboards'].values())
    statuses = {item['family']: item['public_status'] for item in snapshot['special_families']}
    assert statuses == {'weighing_v1': 'implemented', 'device_operation_v1': 'implemented',
                        'centrifuge_v2': 'pending', 'rack_recall_v4': 'pending'}
    assert all(item['status'] == 'implemented' for item in load_release_protocols()['protocols'])


def test_special_assets_have_real_previews_and_bidirectional_links():
    snapshot = build_public_snapshot([])
    objects = {item['object_id']: item for item in snapshot['objects']['objects']}
    with tempfile.TemporaryDirectory() as directory:
        client = create_app(outputs_root=Path(directory)).test_client()
        for object_id in SPECIAL_OBJECTS:
            item = objects[object_id]
            assert item['media'] and item['image_kind'] == 'usd_render'
            assert item['special_families']
            page = client.get(f'/objects/{object_id}/').get_data(as_text=True)
            for family in item['special_families']:
                assert f'/tasks/protocol/{family["protocol_id"]}/' in page
                detail = client.get(f'/tasks/protocol/{family["protocol_id"]}/').get_data(as_text=True)
                assert f'/objects/{object_id}/' in detail
    for name in ('feedback-scale', 'feedback-weight-set', 'feedback-sample'):
        folder = REPOSITORY_ROOT / 'eai4s_eval/configs/assets/derived_feedback'
        source = json.loads((folder / f'{name}.source.json').read_text())
        assert hashlib.sha256((folder / f'{name}.usda').read_bytes()).hexdigest() == source['asset_sha256']
        assert hashlib.sha256((REPOSITORY_ROOT / source['generator']).read_bytes()).hexdigest() == source['generator_sha256']


def test_static_export_includes_special_videos_and_new_assets_but_no_hero():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        export_public_site(output_dir=root/'site', outputs_root=root/'runs',
                           include_public_media=True, base_url='/Labference')
        page = (root/'site/index.html').read_text()
        assert 'id="authors"' not in page and 'href="#authors"' not in page
        assert page.count('data-institution') == 9
        bib = unescape(re.search(r'<pre id="bibtex">(.*?)</pre>', page, re.S).group(1))
        assert bib.count(' and ') == 18
        assert 'Hongwei Shi' in bib
        assert 'Zhuang Xiong' in bib
        assert 'Xiaojing Liu' in bib and 'Chaomin Guo' in bib and 'Feichan Yang' in bib
        for logo in NEW_PARTNER_LOGOS:
            assert (root/'site/static/web_assets/brand'/logo).is_file()
            assert f'/Labference/static/web_assets/brand/{logo}' in page
        for logo in ('objecteye-logo-v2.jpg', 'zidong-taichu-logo-v2.jpg'):
            assert (root/'site/static/web_assets/brand'/logo).is_file()
            assert f'/Labference/static/web_assets/brand/{logo}' in page
        for old_logo in ('objecteye-logo.png', 'zidong-taichu-logo.png'):
            assert not (root/'site/static/web_assets/brand'/old_logo).exists()
        for scene in ('centrifuge-scene.webp', 'rack-recall-scene.webp'):
            assert (root/'site/static/web_assets/task_media/special'/scene).is_file()
        assert '<h2 class="section-title">Abstract</h2>' in page
        assert page.index('id="abstract"') < page.index('id="special-tasks"')
        parser = MediaParser()
        parser.feed(page)
        assert all((root/'site'/reference.removeprefix('/Labference/')).is_file()
                   for reference in parser.references)
        for name in ('weighing-hard-front.mp4', 'device-operation-hard-front.mp4'):
            assert (root/'site/static/web_assets/task_media/special'/name).is_file()
        for object_id in SPECIAL_OBJECTS:
            assert (root/'site/static/web_assets/object_images'/f'{object_id}.webp').is_file()
        assert not (root/'site/static/web_assets/hero/labbench-family-assets-v2.webp').exists()
