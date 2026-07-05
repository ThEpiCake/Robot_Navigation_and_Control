# הגשה סופית — ניווט ובקרת רובוטים (Robot Navigation and Control)

**מגישים:** איתי ברון (209438910), רועי זהבי (209146216)
**מנחה:** פרופ' אמיר שפירא
**קורס:** 362-2-5481, אביב 2026

## תוכן ההגשה

| קובץ | תיאור |
|---|---|
| `Final_Project_Navigation_and_Control_209146216_209438910.pdf` | מסמך מאוחד: דו"ח סופי (עברית) + נספח א' — המצגת שהוצגה בכיתה ב-1.7.2026 (13 שקפים, `FUEL.pptx`) + נספח ב' — עותק המאמר (Zhou et al., *FUEL*, IEEE RA-L 2021) |

המסמך נבנה מ-`../review_main/main.tex` (קומפילציה: `latexmk -xelatex main.tex`).

## קוד המקור (GitHub)

ריפו ציבורי להגשה: <https://github.com/ThEpiCake/Robot_Navigation_and_Control>

- **חלק 1 — זרוע רובוטית:** `src/` (חבילות ROS 2: `my_robot_description`, `my_robot_control`, `my_robot_bringup`)
- **חלק 2 — רחפן אוטונומי (FUEL + VFH):** `swarm_project/ros2_ws/src/` (חבילות `swarm_*`)
- **תוצאות וגרפים:** `results/`
- **המצגת שהוצגה בכיתה:** `FUEL.pptx`
- **עותק המאמר:** `FUEL- Fast UAV Exploration using Incremental Frontier Structure and Hierarchical Planning.pdf`

## סרטוני הסימולציה (כלולים בריפו)

| קובץ | תיאור |
|---|---|
| `../results/arm_gazebo_demo.mp4` | חלק 1 — הזרוע ב-Gazebo וב-RViz (1:12 דק') |
| `../results/drone_swarm_mission_run.mp4` | חלק 2 — הרצה מלאה של משימת החקר האוטונומי (18:25 דק') |

GIFים מקוצרים של שני הסרטונים מוטמעים ב-`README.md` של תקיית הקורס.
