// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 코드 하이라이팅 적용
    hljs.highlightAll();
    
    // 탭 기능 구현
    initTabs();
    
    // 아코디언 기능 구현
    initAccordion();
    
    // 차트 생성
    createProblemTypeChart();
});

// 탭 기능 구현
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // 현재 탭 그룹 찾기
            const tabHeader = this.closest('.tab-header');
            const tabContent = tabHeader.nextElementSibling;
            
            // 같은 그룹의 모든 버튼에서 active 클래스 제거
            tabHeader.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
            });
            
            // 현재 버튼에 active 클래스 추가
            this.classList.add('active');
            
            // 모든 탭 패널 숨기기
            tabContent.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            
            // 선택된 탭 패널 표시
            const targetId = this.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// 아코디언 기능 구현
function initAccordion() {
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    accordionHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const accordionItem = this.parentElement;
            
            // 토글 기능 (열려있으면 닫고, 닫혀있으면 열기)
            accordionItem.classList.toggle('active');
            
            // 아이콘 변경
            const icon = this.querySelector('.accordion-icon');
            if (accordionItem.classList.contains('active')) {
                icon.textContent = '−';
            } else {
                icon.textContent = '+';
            }
        });
    });
}

// 문제 유형 차트 생성
function createProblemTypeChart() {
    const ctx = document.getElementById('problemTypeChart').getContext('2d');
    
    const data = {
        labels: ['데이터 전처리', '탐색적 데이터 분석', '기계학습 모델 구축', '모델 평가', '기초 통계 분석', '통계적 추론', '상관/회귀 분석', '다변량/시계열 분석'],
        datasets: [{
            label: 'ADP 실기 시험 문제 유형 분포',
            data: [25, 15, 20, 15, 10, 5, 5, 5],
            backgroundColor: [
                'rgba(52, 152, 219, 0.7)',
                'rgba(46, 204, 113, 0.7)',
                'rgba(155, 89, 182, 0.7)',
                'rgba(52, 73, 94, 0.7)',
                'rgba(241, 196, 15, 0.7)',
                'rgba(230, 126, 34, 0.7)',
                'rgba(231, 76, 60, 0.7)',
                'rgba(149, 165, 166, 0.7)'
            ],
            borderColor: [
                'rgba(52, 152, 219, 1)',
                'rgba(46, 204, 113, 1)',
                'rgba(155, 89, 182, 1)',
                'rgba(52, 73, 94, 1)',
                'rgba(241, 196, 15, 1)',
                'rgba(230, 126, 34, 1)',
                'rgba(231, 76, 60, 1)',
                'rgba(149, 165, 166, 1)'
            ],
            borderWidth: 1
        }]
    };
    
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    font: {
                        size: 12
                    }
                }
            },
            title: {
                display: true,
                text: 'ADP 실기 시험 문제 유형 분포 (%)',
                font: {
                    size: 16
                }
            }
        }
    };
    
    new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: options
    });
}

// 스크롤 시 네비게이션 바 스타일 변경
window.addEventListener('scroll', function() {
    const nav = document.querySelector('nav');
    if (window.scrollY > 100) {
        nav.style.backgroundColor = 'rgba(44, 62, 80, 0.95)';
        nav.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    } else {
        nav.style.backgroundColor = 'var(--secondary-color)';
        nav.style.boxShadow = 'var(--box-shadow)';
    }
});

// 부드러운 스크롤 기능
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        
        window.scrollTo({
            top: targetElement.offsetTop - 70,
            behavior: 'smooth'
        });
    });
});

// 반응형 메뉴 토글 기능 (모바일 화면용)
function toggleMobileMenu() {
    const navList = document.querySelector('nav ul');
    navList.classList.toggle('show');
}

// 페이지 로드 시 첫 번째 아코디언 항목 자동 열기
window.addEventListener('load', function() {
    // 각 섹션의 첫 번째 아코디언 항목 열기
    const sections = ['machine-learning', 'statistics'];
    
    sections.forEach(section => {
        const firstAccordion = document.querySelector(`#${section} .accordion-item`);
        if (firstAccordion) {
            firstAccordion.classList.add('active');
            const icon = firstAccordion.querySelector('.accordion-icon');
            if (icon) {
                icon.textContent = '−';
            }
        }
    });
});
