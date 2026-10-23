"""Explicit contact profiles for the demo; never modifies training assets."""
import math

PROFILES = ('frictional-body', 'training')


def add_contact_arguments(parser, default='frictional-body'):
    parser.add_argument('--contact-profile', choices=PROFILES, default=default,
                        help='Body-ground friction, or exact training contact settings')
    parser.add_argument('--body-friction', type=float, default=0.5,
                        help='Assumed body sliding coefficient (not hardware calibrated)')


def configure_contacts(root, profile='frictional-body', body_friction=0.5):
    if profile not in PROFILES:
        raise ValueError(f'Unknown contact profile: {profile}')
    if not math.isfinite(body_friction) or body_friction < 0:
        raise ValueError('Body friction must be finite and nonnegative')
    for geom in root.findall('.//worldbody//geom'):
        name = geom.get('name', '')
        if not name.endswith('_collision'):
            continue
        foot = '_foot' in name
        # Preserve training masks and higher robot priority over the floor.
        # condim=1 is frictionless even if the floor specifies nonzero friction.
        geom.attrib.update(contype='0', conaffinity='1',
                           condim='3' if foot or profile == 'frictional-body' else '1')
        if foot:
            geom.set('friction', '0.6 0.005 0.0001')
        elif profile == 'frictional-body':
            geom.set('friction', f'{body_friction} 0.005 0.0001')
    return dict(profile=profile, foot_sliding_friction=0.6,
                body_sliding_friction=body_friction if profile == 'frictional-body' else 0.,
                hardware_calibrated=False)
